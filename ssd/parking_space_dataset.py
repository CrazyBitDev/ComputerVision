import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat


class ParkingSpaceDataset(Dataset):
    """
    Custom dataset for parking space detection.
    """
    def __init__(self, path, device=None):
        super(ParkingSpaceDataset, self).__init__()

        self.device = device if device is not None else torch.device("cpu")
        self.transform = None
        
        image_path = path + "/images"
        label_path = path + "/labels"

        self.image_names = sorted([f for f in os.listdir(image_path) if f.endswith('.png')])

        self.images = []
        self.labels = []
        self.boxes = []

        # default transform for the images (to tensor, normalize)
        self.default_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # for each image in the dataset
        for image_name in self.image_names:

            # load the image, convert it to RGB and apply the default transform,
            image = Image.open(os.path.join(image_path, image_name)).convert('RGB')
            image = self.default_transform(image)

            # add the image to the dataset
            self.images.append(
                image
            )

            # load the corresponding label file
            with open(os.path.join(label_path, image_name.replace('.png', '.txt')), 'r') as f:
                label_lines = f.readlines()

            labels = []
            boxes = []
            # for each line in the label file
            for line in label_lines:
                parts = line.strip().split()
                # parts is in format class_id x1 y1 x2 y2 x3 y3 x4 y4
                # split the line into class and coordinates
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                
                # convert the coordinates to the format (x1, y1, x2, y2)
                x1 = min(coords[0::2])
                y1 = min(coords[1::2])
                x2 = max(coords[0::2])
                y2 = max(coords[1::2])

                # convert the coordinates from relative format to absolute format
                x1 *= image.shape[2]
                y1 *= image.shape[1]
                x2 *= image.shape[2]
                y2 *= image.shape[1]

                # append the class id and the coordinates to the labels and boxes lists
                labels.append(class_id)
                boxes.append([x1, y1, x2, y2])

            labels = np.array(labels)
            boxes = np.array(boxes, dtype=np.float32)

            # convert the labels to a tensor and the boxes to a BoundingBoxes object, add them to the dataset
            labels = torch.tensor(labels, dtype=torch.long).to(self.device)
            boxes = BoundingBoxes(
                boxes, format=BoundingBoxFormat.XYXY, canvas_size=(image.shape[1], image.shape[2]),
                device=self.device
            )

            self.labels.append(labels)
            self.boxes.append(boxes)

    def set_transform(self, transform):
        """
        Set a custom transform for the dataset.
        This transform will be applied to each image, boxes and labels in the dataset, during the __getitem__ call.
        """
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        labels = self.labels[idx]
        boxes = self.boxes[idx]

        # if the transform is set, apply it to the image, boxes and labels
        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)

        # convert boxes to integer format and ensure they are non-negative
        boxes = boxes.int()
        boxes[boxes < 0] = 0

        # ensuring that boxes are valid (width and height > 1)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        valid_mask = (widths > 1) & (heights > 1)

        boxes = boxes[valid_mask]
        labels = labels[valid_mask]

        # construct the image tensor, the label object and return them
        image = image.to(self.device)

        label_object = {
            'boxes': boxes,
            'labels': labels,
        }

        return image, label_object
