from ultralytics import YOLO
from pathlib import Path
import os
import pickle as pkl
import torch

# get list of directories in parking-space-detection-dataset\5-Fold_Cross-val
dataset_path = Path('parking-space-detection-dataset')
dataset_tune_path = dataset_path / "80-20_Split"

model = YOLO('yolov8n.pt', task="detect")
best_parameters = model.tune(
    data=(dataset_tune_path / "dataset.yaml").resolve(),
    batch=16,
    workers=4,
    epochs=30,
    iterations=50,  # number of trials
    optimizer='Adam',  # or SGD
    imgsz=500,
    cache=True,
    val=True,  # enable validation
    
    project="results/yolo/tuning",
    name="tune",

    degrees=90.0,
    perspective=0.001,
    flipud=0.5,

    dropout=0.5,
    plots=False  # optionally turn off plots for speed
)
