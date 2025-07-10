# Violence Detection using Deep Learning

This project focuses on detecting violent content in videos using pre-trained deep learning models. By leveraging transfer learning with image classification architectures, we process video frames to identify whether a clip contains violence.

## Overview

Violence detection is an important task in areas like surveillance, online content moderation, and public safety. This project implements a deep learning pipeline that:

- Extracts frames from videos
- Processes them into a fixed-size batch
- Feeds them through a frozen CNN backbone (e.g., EfficientNetB0, ResNet50)
- Trains a custom classifier on the extracted features

## Key Features

- Uses transfer learning for efficient model training
- Frame-wise analysis with temporal batching (10-frame input)
- Configurable model architecture (plug-and-play base models)
- Visualizations of training performance (loss, accuracy)

## Dependencies

- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV (for video processing)

## Getting Started

1. Clone the repository or download the notebook.
2. Prepare your dataset in a folder structure like:

```
dataset/
├── violence/
│   ├── video1.mp4
│   └── ...
└── nonviolence/
    ├── video2.mp4
    └── ...
```

3. Run the notebook `Violence_detection.ipynb`.

## Model Architecture

The base model is loaded without its top layers and frozen during training. It is wrapped in a Sequential model that includes:

- Input: batch of shape (10, 224, 224, 3)
- Rescaling layer
- TimeDistributed base model
- Global average pooling
- Dense classifier
