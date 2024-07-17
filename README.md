# Driver Drowsiness Detection

This repository contains a Python implementation of a driver drowsiness detection system using OpenCV, MediaPipe, and Keras. The system detects drowsiness by monitoring eye closure and yawning, and it alerts the user if drowsiness is detected.

## Features

- Real-time drowsiness detection using a webcam.
- Eye closure detection using a pre-trained Convolutional Neural Network (CNN) model.
- Yawning detection using a pre-trained Keras model.
- Alerts the user with an audio signal when drowsiness is detected.

## Requirements

- Python 3.6+
- OpenCV
- MediaPipe
- TensorFlow/Keras
- NumPy
- Pygame

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/driver-drowsiness-detection.git
    cd driver-drowsiness-detection
    ```

2. Install the required Python packages:

3. Ensure you have the following files in the project directory:
    - `cnnCat2.h5`: Pre-trained model for eye closure detection.
    - `crazy4.keras`: Pre-trained model for yawning detection.
    - `alert.mp3`: Audio file for alerting the user.

## Usage

Run the `final1.py` script to start the drowsiness detection system:

```sh
python final1.py
