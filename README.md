# Smart Parking Management using Object Detection Models - Computer Vision
Project by Matteo Ingusci

## Motivations
Urbanization and the rapid growth of the number of vehicles have made required the use of a efficient parking management.
Drivers usually waste time searching for an available parking spot, which consequently wastes fuel and causes environmental pollution.

Traditionally smart cities deploy different monitors with the number of available parking spots in certain zones, usually managed by a private company, which requires a toll to the use.
Alternatives are very limited and may require a sensor-based system which is often very expensive since each parking spot requires its own sensor.

The proposed project addresses this problem by training and testing object detection models to build an intelligent parking system.
The idea is to use a real-time video feed from cameras installed in the parks to detect and track occupied and vacant parking slots, to have a real-time analysis.
This system combines flexibility and scalability since it can easily expanded by adding just some other cameras.

The project will explore the application and reliability of certain architecture of Neural Networks, comparing them and analyzing the results.

## Objectives

The goal of this project is to design, implement, and evaluate models that can be used in a smart parking management system, which are able to identify occupied and free parking spaces, using some actual deep learning and computer vision techniques.

The goal is also to analyze and compare different object detection models: YOLOv8 and SSD.
The implementation must consider the training and evaluation on a small dataset, testing the ability of the models to generalize, with the support of data augmentation techniques and parameter search.

## Project structure

- `setup_dataset.ipynb`: jupyter notebook to download, unzip and preprocess the dataset (conversion labels to YOLO, split the data with K-Fold).
- `yolo_tune.py`: find the best hyperparameters for the YOLO model train.
- `ssd_tune.py`: find the best hyperparameters for the SSD model train.
- `yolo_train.py`: train the YOLO models
- `ssd_train.py`: train the SSD models
- `image_gen.ipynb`: load the tensorboard with the training data and draw plots
- `test.py`: test the models on validation sets, save images with detections# ComputerVision
