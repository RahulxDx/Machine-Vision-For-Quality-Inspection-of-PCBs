# Machine Vision for Quality Inspection of PCBs

This project focuses on utilizing machine vision techniques to perform automated quality inspection of Printed Circuit Boards (PCBs). A custom YOLOv8 model was developed to detect various types of faults in PCBs and integrated seamlessly with NVIDIA DeepStream SDK for enhanced performance and deployment capabilities.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Integration with NVIDIA DeepStream](#integration-with-nvidia-deepstream)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Overview
Automated quality inspection in the electronics manufacturing industry is crucial for ensuring product reliability and reducing manual errors. This project leverages computer vision and deep learning to identify defects in PCBs such as missing components, soldering issues, and surface anomalies.

## Features
- Fault detection using a custom-trained YOLOv8 model.
- Real-time inspection capabilities.
- Integration with NVIDIA DeepStream SDK for accelerated inferencing and deployment.
- Scalable and adaptable for various PCB designs.

## Technologies Used
- **Python**: For scripting and model training.
- **YOLOv8**: A state-of-the-art object detection model for fault detection.
- **NVIDIA DeepStream SDK**: For optimized deployment on NVIDIA GPUs.
- **Flask**: For creating a lightweight API for integration.
- **Docker**: For containerized deployment.

## Setup and Installation

### Prerequisites
- NVIDIA GPU with CUDA support
- Docker installed
- Python 3.8 or higher

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/machine-vision-pcb.git
   cd machine-vision-pcb
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Build the Docker image:
   ```bash
   docker build -t pcb-inspection .
   ```
4. Run the container:
   ```bash
   docker run --rm -p 5000:5000 pcb-inspection
   ```

## Usage
1. Access the Flask API at `http://localhost:5000`.
2. Upload PCB images for inspection.
3. The API will return fault detection results with bounding boxes and confidence scores.

## Model Training
The YOLOv8 model was trained on a custom dataset of PCB images annotated with various fault types. Training was conducted in Google Colab with the following steps:

1. Prepare the dataset in YOLO format.
2. Train the model:
   ```bash
   yolo task=detect mode=train data=pcb.yaml model=yolov8n.pt epochs=50 imgsz=640
   ```
3. Export the trained model for inference:
   ```bash
   yolo export format=onnx
   ```

## Integration with NVIDIA DeepStream
1. Convert the YOLOv8 model to TensorRT engine using DeepStream tools.
2. Update the DeepStream configuration files with the model path.
3. Deploy the model using DeepStream:
   ```bash
   deepstream-app -c deepstream_config.txt
   ```

## Results
- **Accuracy**: Achieved high precision and recall in detecting PCB faults.
- **Performance**: Real-time inferencing with low latency using NVIDIA DeepStream.

## Future Enhancements
- Expand the dataset to include more PCB fault types.
- Add support for multi-camera setups.
- Explore edge deployment using NVIDIA Jetson devices.

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request with your enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
