from pathlib import Path
import os
import pickle as pkl
import torch
import yaml

from ssd.model import construct_ssd_model
from ssd.parking_space_dataset import ParkingSpaceDataset

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

from torchvision.transforms import v2

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from tqdm import trange

from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.ops import box_iou

def collate_fn(batch):
    """
    Custom collate function to handle batches of images and targets.
    """
    # Unzip the batch into images and targets
    images, targets = zip(*batch)
    # Return them as lists
    return list(images), list(targets)

def calculate_metrics(preds, targets, iou_threshold=0.5, score_threshold=0.5):
    """
    Calculate custom metrics for object detection tasks.
    """
    TP = 0
    FP = 0
    FN = 0
    total_iou = 0.0
    iou_count = 0

    # for each prediction and target pair
    for pred, target in zip(preds, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        pred_labels = pred['labels']

        gt_boxes = target['boxes']
        gt_labels = target['labels']

        # Filter predictions by score threshold
        keep = pred_scores >= score_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]

        # if background prediction, increemnt FP and FN, skip to the next
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            FP += len(pred_boxes)
            FN += len(gt_boxes)
            continue

        # Calculate IoU between predictions and ground truth boxes using torchvision's box_iou
        ious = box_iou(pred_boxes, gt_boxes)  # [num_preds, num_gt]

        matched_pred_idx = set()
        matched_gt_idx = set()

        # for each predicted box
        for i in range(len(pred_boxes)):
            best_iou = 0
            best_j = -1
            # for each ground truth box
            for j in range(len(gt_boxes)):
                if j in matched_gt_idx:
                    continue
                iou = ious[i, j].item()
                # find the best IoU that is above the threshold
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_j = j

            # if a match is found, increment TP, add to matched indices and update total IoU
            if best_j >= 0:
                TP += 1
                matched_pred_idx.add(i)
                matched_gt_idx.add(best_j)
                total_iou += best_iou
                iou_count += 1
            else:
                # no match found, increment FP
                FP += 1

        # for each ground truth box that was not matched, increment FN
        FN += len(gt_boxes) - len(matched_gt_idx)

    # Calculate precision, recall, and mean IoU
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    mean_iou = total_iou / (iou_count + 1e-6)

    return {
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou
    }


metrics = {}

dataset_path = Path('parking-space-detection-dataset')
dataset_tune_results = Path('results') / "ssd" / "tuning"
dataset_training_results = Path('results') / "ssd" / "training"
dataset_cross_val_path = dataset_path / "4-Fold_Cross-val"
splits = os.listdir(dataset_cross_val_path)

# Load the best hyperparameters from the tuning results
with open(dataset_tune_results / "best_hyperparameters.yaml") as f:
    best_parameters = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 500

# create the transformation pipeline for data augmentation using the best hyperparameters
transform = v2.Compose(
    [
        v2.RandomHorizontalFlip(
            p=best_parameters["horizontal_flip_p"]
        ),
        v2.RandomVerticalFlip(
            p=best_parameters["vertical_flip_p"]
        ),
        v2.RandomAffine(
            degrees=best_parameters["affine_degrees"],
            translate=(best_parameters["translate"],) * 2,
            scale=(
                best_parameters["scale_min"],
                best_parameters["scale_max"]
            ),
            shear=best_parameters["shear"],
        ),
        v2.ColorJitter(
            brightness=best_parameters["brightness"],
            contrast=best_parameters["contrast"],
            saturation=best_parameters["saturation"],
            hue=best_parameters["hue"],
        ),
        v2.RandomPerspective(
            distortion_scale=best_parameters["perspective_distortion"],
            p=best_parameters["perspective_p"]
        ),
        v2.GaussianBlur(
            kernel_size=3,
            sigma=best_parameters["blur_sigma"]
        ),
        v2.RandomGrayscale(p=best_parameters["grayscale_p"]),
        v2.RandomErasing(
            p=best_parameters["erase_p"],
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3)
        ),
    ],
)

# Initialize the metric for evaluation
metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])

# for each split
for split in splits:

    # create the model and move it to the device
    model = construct_ssd_model()
    model.to(device)

    split_dir = dataset_training_results / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # create a SummaryWriter for TensorBoard logging
    writer = SummaryWriter(
        log_dir=split_dir
    )

    # Load the best hyperparameters
    lr = best_parameters['lr'] # best learning rate

    # best optimizer
    if best_parameters['optimizer'] == 'Adam':
        optimizer = Adam(
            model.parameters(),
            lr=lr,
            weight_decay=best_parameters['weight_decay'] # best weight decay
        )
    elif best_parameters['optimizer'] == 'SGD':
        optimizer = SGD(
            model.parameters(),
            lr=lr,
            momentum=best_parameters['momentum'], # best momentum
            weight_decay=best_parameters['weight_decay'] # best weight decay
        )

    # best warmup epochs
    warmup_epochs = best_parameters['warmup_epochs']

    # best scheduler
    if best_parameters['scheduler'] == 'CosineAnnealingLR':
        T_max = best_parameters['T_max']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif best_parameters['scheduler'] == 'StepLR':
        step_size = best_parameters['step_size']
        gamma = best_parameters['gamma']
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Create the dataset and dataloaders for the split
    train_dataset = ParkingSpaceDataset(
        str(dataset_cross_val_path / splits[0] / "train"),
        device=device
    )
    val_dataset = ParkingSpaceDataset(
        str(dataset_cross_val_path / splits[0] / "val"),
        device=device
    )

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
    # set the data augmentation transform for the training dataset
    train_dataset.set_transform(transform)
    
    # for each epoch
    for epoch in trange(num_epochs, leave=False, desc=f"Training on split {split}"):

        # set the model to training mode
        model.train()

        # if in the warm-up phase, adjust the learning rate
        if epoch < warmup_epochs:
            # Linear warm-up
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_factor * lr
        else:
            scheduler.step()  # Step only after warm-up

        # for each batch in the training dataloader
        for images, targets in train_dataloader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # forward pass, calculate losses and backpropagate
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

        writer.add_scalar("Loss/train", losses.item(), epoch)

        # validation phase, set the model to evaluation mode
        model.eval()
        metric.reset()

        preds, targets = [], []
        with torch.no_grad():
            # for each batch in the validation dataloader
            for images, target in val_dataloader:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # forward pass, get the outputs
                outputs = model(images)

                metric.update(outputs, target)
                preds.extend(outputs)
                targets.extend(target)
                
        # compute the metrics
        results = metric.compute()

        # log the results to TensorBoard
        writer.add_scalar("mAP/val", results['map'].item(), epoch)
        writer.add_scalar("mAP/val_50", results['map_50'].item(), epoch)

        # calculate custom metrics
        custom_metrics = calculate_metrics(preds, targets, iou_threshold=0.5, score_threshold=0.5)

        # precision, recall
        writer.add_scalar("Precision/val", custom_metrics['precision'], epoch)
        writer.add_scalar("Recall/val", custom_metrics['recall'], epoch)
        writer.add_scalar("IoU/val", custom_metrics['mean_iou'], epoch)

        writer.flush()


    # save models in split_dir
    model_save_path = split_dir / "model.pth"
    torch.save(model.state_dict(), model_save_path)



