# Object Detection using Webcam

This repository contains a real-time object detection application using a webcam. The project uses the MobileNetSSD model with OpenCV's Deep Neural Network (DNN) module to detect and classify objects in a video stream.

## Features
- Real-time object detection using a webcam
- Pre-trained MobileNetSSD model for object recognition
- Bounding boxes and confidence scores displayed on detected objects
- Easy setup and execution

## Requirements
Make sure you have Python installed, then install the necessary dependencies using:

```bash
pip install -r requirements.txt
```

## Installation & Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Object-Detection-Webcam.git
   cd Object-Detection-Webcam
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python "Object Detection.py"
   ```

## Files in the Repository
- `Object Detection.py` - Main script to run object detection.
- `MobileNetSSD_deploy.prototxt` - Model configuration file.
- `MobileNetSSD_deploy.caffemodel` - Pre-trained model weights.
- `requirements.txt` - List of dependencies.
- `.gitignore` - Ignore unnecessary files.

## Object Classes Detected
The MobileNetSSD model detects the following objects:
- Aeroplane, Bicycle, Bird, Boat, Bottle, Bus, Car, Cat, Chair, Cow, Dining Table, Dog, Horse, Motorbike, Person, Potted Plant, Sheep, Sofa, Train, Monitor, Mouse.

## Keyboard Controls
- **Press 'q'** to exit the application.

## Notes
- Ensure your webcam is connected or use an IP camera (modify the `cap = cv2.VideoCapture()` line accordingly).
- Adjust confidence thresholds in the script if needed.

## License
This project is licensed under the MIT License.

