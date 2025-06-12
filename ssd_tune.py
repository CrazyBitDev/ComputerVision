import os
import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights, _utils
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.transforms import v2

import matplotlib.pyplot as plt

import optuna
import optuna.visualization.matplotlib as vis

import pandas as pd

from tqdm import trange
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import yaml

from ssd.parking_space_dataset import ParkingSpaceDataset
from ssd.model import construct_ssd_model

def collate_fn(batch):
    """
    Custom collate function to handle batches of images and targets.
    """
    # Unzip the batch into images and targets
    images, targets = zip(*batch)
    # Return them as lists
    return list(images), list(targets)

def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning of SSD model.
    This function constructs the SSD model with hyperparameters suggested by Optuna,
    """

    model = construct_ssd_model()
    model = model.to(device)

    # --- Hyperparameters ---
    # Learning rate, weight decay, and optimizer type
    # These hyperparameters are suggested by Optuna    
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    wd = trial.suggest_float("weight_decay", 0.0, 1e-3)

    # Choose an optimizer
    opt_type = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
    if opt_type == "SGD":
        # SGD requires momentum
        momentum = trial.suggest_float("momentum", 0.5, 0.9)
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    elif opt_type == "Adam":
        # Adam utilizes common paarameters
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Number of warmup epochs
    warmup_epochs = trial.suggest_int("warmup_epochs", 1, 5)

    # Choose a scheduler
    scheduler_type = trial.suggest_categorical("scheduler", ["CosineAnnealingLR", "StepLR"])
    if scheduler_type == "CosineAnnealingLR":
        # Cosine Annealing requires T_max (max number of iterations)
        T_max = trial.suggest_int("T_max", 10, 50)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == "StepLR":
        # StepLR requires step_size and gamma
        step_size = trial.suggest_int("step_size", 5, 30)
        gamma = trial.suggest_float("gamma", 0.1, 0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Data augmentation transformer with hyperparameters suggested by Optuna
    transform = v2.Compose(
        [
            v2.RandomHorizontalFlip(p=trial.suggest_float("horizontal_flip_p", 0.0, 1.0)),
            v2.RandomVerticalFlip(p=trial.suggest_float("vertical_flip_p", 0.0, 1.0)),
            v2.RandomAffine(
                degrees=trial.suggest_int("affine_degrees", 0, 45),
                translate=(trial.suggest_float("translate", 0.0, 0.3),) * 2,
                scale=(trial.suggest_float("scale_min", 0.7, 1.0),
                    trial.suggest_float("scale_max", 1.0, 1.3)),
                shear=trial.suggest_float("shear", 0.0, 20.0),
            ),
            v2.ColorJitter(
                brightness=trial.suggest_float("brightness", 0.0, 0.5),
                contrast=trial.suggest_float("contrast", 0.0, 0.5),
                saturation=trial.suggest_float("saturation", 0.0, 0.5),
                hue=trial.suggest_float("hue", 0.0, 0.2),
            ),
            v2.RandomPerspective(
                distortion_scale=trial.suggest_float("perspective_distortion", 0.1, 0.6),
                p=trial.suggest_float("perspective_p", 0.0, 0.5)
            ),
            v2.GaussianBlur(
                kernel_size=3,
                sigma=trial.suggest_float("blur_sigma", 0.1, 2.0)
            ),
            v2.RandomGrayscale(p=trial.suggest_float("grayscale_p", 0.0, 0.3)),
            v2.RandomErasing(
                p=trial.suggest_float("erase_p", 0.0, 0.5),
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3)
            ),
        ],
    )

    # Set the transform for the training dataset
    train_dataset.set_transform(transform)

    num_epochs = 100
    metric = MeanAveragePrecision(iou_thresholds=[0.5])

    model.train() 
    # Training loop
    # Iterate over epochs
    for epoch in trange(num_epochs, desc=f"Tuning {trial.number}", leave=False):

        
        # if in the warm-up phase, adjust the learning rate
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_factor * lr
        else:
            scheduler.step()  # Step only after warm-up
        
        # Iterate over batches in the training dataloader
        for images, targets in train_dataloader:
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Validation loop, update the metric with predictions from the validation set
    model.eval()
    with torch.no_grad():
        for images, targets in val_dataloader:
            preds = model(images)
            metric.update(preds, targets)

    # Compute the mean average precision (mAP) from the metric and return it as the objective value
    results = metric.compute()
    return results['map'].item()
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ParkingSpaceDataset("./parking-space-detection-dataset/80-20_Split/train", device=device)
val_dataset = ParkingSpaceDataset("./parking-space-detection-dataset/80-20_Split/val", device=device)

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=4,
    collate_fn=collate_fn,
)

val_dataloader = DataLoader(
    val_dataset,
    shuffle=False,
    batch_size=4,
    collate_fn=collate_fn,
)

# Create an Optuna study and optimize the objective function
# the goal is to maximize the mean average precision (mAP)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# print and save data
print("Best trial:")
print(f"  Value: {study.best_value:.4f}")

if not os.path.exists("results/ssd/tuning"):
    os.makedirs("results/ssd/tuning")

with open("results/ssd/tuning/best_hyperparameters.yaml", "w") as f:
    yaml.dump(study.best_trial.params, f)

df = study.trials_dataframe()
df.to_csv("results/ssd/tuning/tune_results.csv", index=False)

fig = vis.plot_optimization_history(study)
fig.figure.savefig("results/ssd/tuning/optimization_history.png")

fig = vis.plot_param_importances(study)
fig.figure.savefig("results/ssd/tuning/param_importances.png")

fig = vis.plot_slice(study)
fig[0].figure.savefig("results/ssd/tuning/slice_plot.png")