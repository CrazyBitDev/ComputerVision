from ultralytics import YOLO
from pathlib import Path
import os
import pickle as pkl
import torch
import yaml

metrics = {}

dataset_path = Path('parking-space-detection-dataset')
dataset_tune_results = Path('results') / "yolo" / "tuning" / "tune"
dataset_cross_val_path = dataset_path / "4-Fold_Cross-val"
splits = os.listdir(dataset_cross_val_path)

# load the best hyperparameters from the tuning results
with open(dataset_tune_results / "best_hyperparameters.yaml") as f:
    best_parameters = yaml.safe_load(f)

# for each split in the cross-validation dataset
for split in splits:
    # load the best hyperparameters from the tuning results and train the model
    model = YOLO('yolov8n.pt', task="detect")
    model.train(
        data=(dataset_cross_val_path / split / "dataset.yaml").resolve(),
        batch=16,
        workers=4,
        epochs=100,
        optimizer='Adam',  # or SGD
        imgsz=500,
        cache=True,
        val=True,  # enable validation
        
        project="results/yolo/training",
        name=split,

        dropout=0.5,
        **best_parameters
    )
    # save the model
    metrics[split] = model.metrics

    del model
    torch.cuda.empty_cache()


def metric_to_dict(metric):
    """
    Save all the YOLO metrics as a dictionary.
    """
    return {
        "curves": metric.curves,
        "curves_results": metric.curves_results,
        "fitness": metric.fitness,
        "keys": metric.keys,
        "maps": metric.maps,
        "names": metric.names,
        "ap_class_index": metric.ap_class_index,
        "result_dict": metric.results_dict,
        "speed": metric.speed,
        "box": {
            "all_ap": metric.box.all_ap,
            "ap": metric.box.ap,
            "ap50": metric.box.ap50,
            "ap_class_index": metric.box.ap_class_index,
            "curves": metric.box.curves,
            "curves_results": metric.box.curves_results,
            "f1": metric.box.f1,
            "f1_curve": metric.box.f1_curve,
            "map": metric.box.map,
            "map50": metric.box.map50,
            "map75": metric.box.map75,
            "maps": metric.box.maps,
            "mp": metric.box.mp,
            "mr": metric.box.mr,
            "nc": metric.box.nc,
            "p": metric.box.p,
            "p_curve": metric.box.p_curve,
            "prec_values": metric.box.prec_values,
            "px": metric.box.px,
            "r": metric.box.r,
            "r_curve": metric.box.r_curve,
        }
    }

# Save the metrics for each split in a dictionary and pickle it
metrics_data = {}
for split, metric in metrics.items():
    metrics_data[split] = metric_to_dict(metric)

with open("metrics_kfold_4_200_16_03.pkl", "wb") as f:
    pkl.dump(metrics_data, f)