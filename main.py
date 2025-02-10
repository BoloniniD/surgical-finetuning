import csv
import itertools
import os
import time
from collections import defaultdict
import copy
import pathlib
from datetime import date
import hydra
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
import wandb
from loguru import logger
from huggingface_hub import hf_hub_download

import utils
from dataset import get_loaders

import warnings
warnings.filterwarnings("ignore")


@torch.no_grad()
def test(model, loader, criterion, cfg):
    """
    Evaluate model performance on test/validation data
    Args:
        model: Neural network model
        loader: DataLoader containing test/val data
        criterion: Loss function
        cfg: Configuration object
    Returns:
        acc: Accuracy on the dataset
        total_loss: Average loss on the dataset
    """
    model.eval()
    all_test_corrects = []
    total_loss = 0.0
    for x, y in loader:
        x, y = x.cuda(), y.cuda()  # Move data to GPU
        logits = model(x)
        loss = criterion(logits, y)
        all_test_corrects.append(torch.argmax(logits, dim=-1) == y)
        total_loss += loss
    acc = torch.cat(all_test_corrects).float().mean().detach().item()
    total_loss = total_loss / len(loader)
    total_loss = total_loss.detach().item()
    return acc, total_loss


def get_lr_weights(model, loader, cfg):
    """
    Calculate learning rate weights for different layers based on gradients
    Args:
        model: Neural network model
        loader: DataLoader containing training data
        cfg: Configuration object
    Returns:
        average_metrics: Dictionary containing average gradient metrics for each layer
    """
    # Get names of non-batch-norm layers
    layer_names = [n for n, _ in model.named_parameters() if "bn" not in n]
    metrics = defaultdict(list)
    average_metrics = defaultdict(float)
    # Only use first 5 batches for gradient calculation
    partial_loader = itertools.islice(loader, 5)
    xent_grads, entropy_grads = [], []

    # Calculate gradients for cross-entropy loss
    for x, y in partial_loader:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss_xent = F.cross_entropy(logits, y)
        grad_xent = torch.autograd.grad(
            outputs=loss_xent, inputs=model.parameters(), retain_graph=True
        )
        xent_grads.append([g.detach() for g in grad_xent])

    def get_grad_norms(model, grads, cfg):
        """Helper function to compute gradient norms"""
        _metrics = defaultdict(list)
        grad_norms, rel_grad_norms = [], []
        for (name, param), grad in zip(model.named_parameters(), grads):
            if name not in layer_names:
                continue
            if cfg.args.auto_tune == "eb-criterion":
                # Compute evidence-based criterion
                tmp = (grad * grad) / (torch.var(grad, dim=0, keepdim=True) + 1e-8)
                _metrics[name] = tmp.mean().item()
            else:
                # Compute relative gradient norm
                _metrics[name] = torch.norm(grad).item() / torch.norm(param).item()
        return _metrics

    # Average metrics across batches
    for xent_grad in xent_grads:
        xent_grad_metrics = get_grad_norms(model, xent_grad, cfg)
        for k, v in xent_grad_metrics.items():
            metrics[k].append(v)
    for k, v in metrics.items():
        average_metrics[k] = np.array(v).mean(0)
    return average_metrics


def train(model, loader, criterion, opt, cfg, orig_model=None):
    """
    Train the model for one epoch
    Args:
        model: Neural network model
        loader: DataLoader containing training data
        criterion: Loss function
        opt: Optimizer
        cfg: Configuration object
        orig_model: Original model for reference (optional)
    Returns:
        acc: Training accuracy
        total_loss: Average training loss
        magnitudes: Dictionary of gradient magnitudes
    """
    all_train_corrects = []
    total_loss = 0.0
    magnitudes = defaultdict(float)

    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = criterion(logits, y)
        all_train_corrects.append(torch.argmax(logits, dim=-1) == y)
        total_loss += loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    acc = torch.cat(all_train_corrects).float().mean().detach().item()
    total_loss = total_loss / len(loader)
    total_loss = total_loss.detach().item()
    return acc, total_loss, magnitudes


@hydra.main(config_path="./config", config_name="config")
def main(cfg):
    # Set up logging directory with current date
    cfg.args.log_dir = pathlib.Path.cwd()
    cfg.args.log_dir = os.path.join(
        cfg.args.log_dir,
        "results",
        cfg.data.dataset_name,
        date.today().strftime("%Y.%m.%d"),
        cfg.args.auto_tune,
    )
    logger.info(f"Log dir: {cfg.args.log_dir}")
    os.makedirs(cfg.args.log_dir, exist_ok=True)

    # Define which layers can be tuned based on model architecture
    #tune_options = [
    #    "first_two_block",
    #    "second_block",
    #    "third_block",
    #    "last",
    #    "all",
    #]
    tune_options = ["all"]
    # Add extra block option for ImageNet-C dataset
    if cfg.data.dataset_name == "imagenet-c":
        tune_options.append("fourth_block")
    # If auto-tuning is enabled, only use 'all' layers option
    if cfg.args.auto_tune != "none":
        tune_options = ["all"]
    # If epochs is 0, only use 'all' layers option
    if cfg.args.epochs == 0:
        tune_options = ["all"]

    # Get corruption types from config
    corruption_types = cfg.data.corruption_types

    # Iterate through each corruption type
    for corruption_type in corruption_types:
        # Set up wandb experiment name
        cfg.wandb.exp_name = f"{cfg.data.dataset_name}_corruption{corruption_type}"
        if cfg.wandb.use:
            utils.setup_wandb(cfg)

        # Set random seed for reproducibility
        utils.set_seed_everywhere(cfg.args.seed)

        # Get data loaders for current corruption type and severity
        loaders = get_loaders(cfg, corruption_type, cfg.data.severity)

        # Iterate through each tuning option (which layers to fine-tune)
        for tune_option in tune_options:
            # Initialize metrics dictionary
            tune_metrics = defaultdict(list)

            # Define grid of learning rates and weight decay values to try
            lr_wd_grid = [
                (1e-1, 1e-4),
                (1e-3, 1e-4),
                (1e-5, 1e-4),
            ]

            # Try each combination of learning rate and weight decay
            for lr, wd in lr_wd_grid:
                # Determine dataset name for model loading
                dataset_name = (
                    "imagenet"
                    if cfg.data.dataset_name == "imagenet-c"
                    else cfg.data.dataset_name
                )

                # Load pre-trained model
                model = load_model(
                    cfg.data.model_name,
                    cfg.user.ckpt_dir,
                    dataset_name,
                    ThreatModel.corruptions.value,
                )
                
                '''model_path = hf_hub_download("edadaltocg/resnet18_cifar10", filename="pytorch_model.bin")
                model = models.resnet18(pretrained=False, num_classes=10)
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                state_dict = torch.load(model_path, map_location=torch.device('cuda'))
                model.load_state_dict(state_dict, strict=False)'''

                #model = timm.create_model("resnet18_cifar10", pretrained=True)

                # Create a copy of original model and move to GPU
                orig_model = copy.deepcopy(model)
                model = model.cuda()

                # Define which parameters to tune based on model architecture and tune_option
                if cfg.data.dataset_name == "cifar10":
                    tune_params_dict = {
                        "all": [model.parameters()],
                        "first_two_block": [
                            model.conv1.parameters(),
                            model.block1.parameters(),
                        ],
                        "second_block": [
                            model.block2.parameters(),
                        ],
                        "third_block": [
                            model.block3.parameters(),
                        ],
                        "last": [model.fc.parameters()],
                    }
                elif cfg.data.dataset_name == "imagenet-c":
                    tune_params_dict = {
                        "all": [model.model.parameters()],
                        "first_second": [
                            model.model.conv1.parameters(),
                            model.model.layer1.parameters(),
                            model.model.layer2.parameters(),
                        ],
                        "first_two_block": [
                            model.model.conv1.parameters(),
                            model.model.layer1.parameters(),
                        ],
                        "second_block": [
                            model.model.layer2.parameters(),
                        ],
                        "third_block": [
                            model.model.layer3.parameters(),
                        ],
                        "fourth_block": [
                            model.model.layer4.parameters(),
                        ],
                        "last": [model.model.fc.parameters()],
                    }

                # Flatten parameter list and create optimizer
                params_list = list(itertools.chain(*tune_params_dict[tune_option]))
                opt = optim.Adam(params_list, lr=lr, weight_decay=wd)

                # Calculate number of trainable parameters
                N = sum(p.numel() for p in params_list if p.requires_grad)

                # print training configuration
                logger.info(
                    f"\nTrain mode={cfg.args.train_mode}, using {cfg.args.train_n} corrupted images for training"
                )
                logger.info(
                    f"Re-training {tune_option} ({N} params). lr={lr}, wd={wd}. Corruption {corruption_type}"
                )

                # Set up loss function and initialize layer weights
                criterion = F.cross_entropy
                
                layer_weights = [
                    0 for layer, _ in model.named_parameters() if "bn" not in layer
                ]
                layer_names = [
                    layer for layer, _ in model.named_parameters() if "bn" not in layer
                ]

                # Training loop
                for epoch in range(1, cfg.args.epochs + 1):
                    # Set model to train mode if specified
                    if cfg.args.train_mode == "train":
                        model.train()

                    # Handle different auto-tuning methods
                    if cfg.args.auto_tune != "none":
                        if cfg.args.auto_tune == "RGN":
                            # Relative Gradient Norm auto-tuning
                            weights = get_lr_weights(model, loaders["train"], cfg)
                            max_weight = max(weights.values())
                            for k, v in weights.items():
                                weights[k] = v / max_weight
                            layer_weights = [
                                sum(x) for x in zip(layer_weights, weights.values())
                            ]
                            tune_metrics["layer_weights"] = layer_weights
                            params = defaultdict()
                            for n, p in model.named_parameters():
                                if "bn" not in n:
                                    params[n] = p
                            params_weights = []
                            for param, weight in weights.items():
                                params_weights.append(
                                    {"params": params[param], "lr": weight * lr}
                                )
                            opt = optim.Adam(params_weights, lr=lr, weight_decay=wd)

                        elif cfg.args.auto_tune == "eb-criterion":
                            # Evidence-based criterion auto-tuning
                            weights = get_lr_weights(model, loaders["train"], cfg)
                            logger.info(
                                f"Epoch {epoch}, autotuning weights {min(weights.values()), max(weights.values())}"
                            )
                            tune_metrics["max_weight"].append(max(weights.values()))
                            tune_metrics["min_weight"].append(min(weights.values()))
                            logger.info(weights.values())
                            for k, v in weights.items():
                                weights[k] = 0.0 if v < 0.95 else 1.0
                            logger.info(f"weight values {weights.values()}")
                            layer_weights = [
                                sum(x) for x in zip(layer_weights, weights.values())
                            ]
                            tune_metrics["layer_weights"] = layer_weights
                            params = defaultdict()
                            for n, p in model.named_parameters():
                                if "bn" not in n:
                                    params[n] = p
                            params_weights = []
                            for k, v in params.items():
                                if k in weights.keys():
                                    params_weights.append(
                                        {"params": params[k], "lr": weights[k] * lr}
                                    )
                                else:
                                    params_weights.append(
                                        {"params": params[k], "lr": 0.0}
                                    )
                            opt = optim.Adam(params_weights, lr=lr, weight_decay=wd)

                        else:
                            # Log fraction of parameters being tuned
                            no_weight = 0
                            for elt in params_weights:
                                if elt["lr"] == 0.0:
                                    no_weight += elt["params"][0].flatten().shape[0]
                            total_params = sum(p.numel() for p in model.parameters())
                            tune_metrics["frac_params"].append(
                                (total_params - no_weight) / total_params
                            )
                            logger.info(
                                f"Tuning {(total_params-no_weight)} out of {total_params} total"
                            )

                    # Train and evaluate model
                    acc_tr, loss_tr, grad_magnitudes = train(
                        model,
                        loaders["train"],
                        criterion,
                        opt,
                        cfg,
                        orig_model=orig_model,
                    )
                    acc_te, loss_te = test(model, loaders["test"], criterion, cfg)
                    acc_val, loss_val = test(model, loaders["val"], criterion, cfg)

                    # Record metrics
                    tune_metrics["acc_train"].append(acc_tr)
                    tune_metrics["acc_val"].append(acc_val)
                    tune_metrics["acc_te"].append(acc_te)

                    # Prepare logging dictionary
                    log_dict = {
                        f"{tune_option}/train/acc": acc_tr,
                        f"{tune_option}/train/loss": loss_tr,
                        f"{tune_option}/val/acc": acc_val,
                        f"{tune_option}/val/loss": loss_val,
                        f"{tune_option}/test/acc": acc_te,
                        f"{tune_option}/test/loss": loss_te,
                    }
                    logger.info(
                        f"Epoch {epoch:2d} Train acc: {acc_tr:.4f}, Val acc: {acc_val:.4f}"
                    )

                    # Log to wandb if enabled
                    if cfg.wandb.use:
                        wandb.log(log_dict)

                # Record learning rate and weight decay
                tune_metrics["lr_tested"].append(lr)
                tune_metrics["wd_tested"].append(wd)

            # Find best performing configuration based on validation accuracy
            best_run_idx = np.argmax(np.array(tune_metrics["acc_val"]))
            best_testacc = tune_metrics["acc_te"][best_run_idx]
            best_lr_wd = best_run_idx // (cfg.args.epochs)

            logger.info(
                f"Best epoch: {best_run_idx % (cfg.args.epochs)}, Test Acc: {best_testacc}"
            )

            # Prepare results for CSV logging
            data = {
                "corruption_type": corruption_type,
                "train_mode": cfg.args.train_mode,
                "tune_option": tune_option,
                "auto_tune": cfg.args.auto_tune,
                "train_n": cfg.args.train_n,
                "seed": cfg.args.seed,
                "lr": tune_metrics["lr_tested"][best_lr_wd],
                "wd": tune_metrics["wd_tested"][best_lr_wd],
                "val_acc": tune_metrics["acc_val"][best_run_idx],
                "best_testacc": best_testacc,
            }

            # Write results to CSV file with retry mechanism
            recorded = False
            fieldnames = data.keys()
            csv_file_name = f"{cfg.args.log_dir}/results_seed{cfg.args.seed}.csv"
            write_header = True if not os.path.exists(csv_file_name) else False
            while not recorded:
                try:
                    with open(csv_file_name, "a") as f:
                        csv_writer = csv.DictWriter(
                            f, fieldnames=fieldnames, restval=0.0
                        )
                        if write_header:
                            csv_writer.writeheader()
                        csv_writer.writerow(data)
                    recorded = True
                except:
                    time.sleep(5)


if __name__ == "__main__":
    main()
