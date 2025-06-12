from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from torchvision import transforms

from ultralytics import YOLO

from ssd.model import construct_ssd_model

def get_yolo_model(results_path, split):
    """
    Load the YOLO model
    """
    yolo_directory = results_path / "yolo" / "training" / split / "weights"
    model = YOLO(yolo_directory / "best.pt", task="detect")
    return model

def get_ssd_model(results_path, split, device=None):
    """
    Load the SSD model
    """
    ssd_directory = results_path / "ssd" / "training" / split
    model = construct_ssd_model()
    # load the weights
    model.load_state_dict(torch.load(ssd_directory / "model.pth", map_location=device))
    model.eval()

    if device is not None:
        model.to(device)

    return model

def plot_detections(image, detections, class_names, colors, file_name):
    """
    Plot the detections on the image and save it

    Args:
        image: PIL Image object
        detections: Tensor of shape (N, 6) where each row is [x1, y1, x2, y2, score, class]
        class_names: List of class names
        colors: List of colors for each class
        file_name: Path to save the output image
    """
   
    # Create the figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    # for each detection
    for det in detections:
        # obtain the coordinates, score and class
        x1, y1, x2, y2, score, cls = det.tolist()
        width, height = x2 - x1, y2 - y1

        # Create a semi-transparent filled rectangle, with the color depending on the class
        rect = patches.Rectangle(
            (x1, y1), width, height,
            edgecolor="black",
            linewidth=1,
            facecolor=colors[int(cls)],
            alpha=0.3
        )
        ax.add_patch(rect)

        # Label text
        cls_name = class_names[int(cls)]
        label = f"{cls_name} {score:.2f}"
        ax.text(x1, y1 - 5, label,
                fontsize=10,
                color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    # remove axis and save the figure
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)

def main():
    # define paths and parameters (like the split and kFold)
    kFold = 4
    split = "split_2"
    results_path = Path('results')
    dataset_path = Path('parking-space-detection-dataset')
    subdataset_path = dataset_path / f"{kFold}-Fold_Cross-val" / split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the YOLO and SSD models
    yolo_model = get_yolo_model(results_path, split)
    ssd_model = get_ssd_model(results_path, split, device)


    # take 5 random images from the validation set of the split
    val_dataset = subdataset_path / "val" / "images"
    val_images = list(val_dataset.glob("*.png"))[:5]

    # Define the transformation for the SSD model (convert the PIL to tensor and normalize it)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Example for ImageNet
                            std=[0.229, 0.224, 0.225])
    ])

    # for each image
    for image_path in val_images:
        # load the image
        image = Image.open(image_path)
        # convert the image in tensor, normalize it and add a batch dimension
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Run the models
        yolo_results = yolo_model(image)
        ssd_results = ssd_model(image_tensor)

        # Extract boxes from YOLO results
        yolo_boxes = yolo_results[0].boxes
        yolo_detections = torch.cat([yolo_boxes.xyxy, yolo_boxes.conf.unsqueeze(1), yolo_boxes.cls.unsqueeze(1)], dim=1)

        # Extract boxes from SSD results
        ssd_boxes = ssd_results[0]['boxes']
        ssd_scores = ssd_results[0]['scores']
        ssd_labels = ssd_results[0]['labels'] - 1 # It is needed because the SSD model consider background as class 0
        ssd_detections = torch.cat([
            ssd_boxes,
            ssd_scores.unsqueeze(1),
            ssd_labels.unsqueeze(1).float()
        ], dim=1)

        # Plot
        plot_detections(
            image, yolo_detections,
            ["occupied", "free"],
            ["blue", "green"],
            "results/test/yolo_predictions_" + image_path.name
        )
        plot_detections(
            image, ssd_detections,
            ["occupied", "free"],
            ["blue", "green"],
            "results/test/ssd_predictions_" + image_path.name
        )


if __name__ == "__main__":
    main()