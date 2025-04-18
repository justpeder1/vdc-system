# 🚗 Vehicle Detection and Counting System

## 📑 Introduction

In modern cities, traffic management and vehicle flow analysis are becoming increasingly critical. With the rising number of vehicles and complex traffic systems, there is a growing need for intelligent and efficient solutions.

This project aims to develop a real-time vehicle detection and counting system using artificial intelligence technologies. Powered by deep learning, this system accurately detects and classifies various types of vehicles, providing an advanced tool for traffic monitoring and smart city initiatives.

To learn more about how this system works and the technical details behind it, you can check out the original article on [Medium](https://medium.com/@gokhakan/vehicle-detection-and-counting-system-building-an-ai-powered-system-for-intelligent-traffic-analysis-8dd07b4ab7ea).

## 🔍 Project Overview

This project provides an advanced AI-powered system for detecting and analyzing vehicles in images and videos. The system utilizes deep learning techniques to identify various vehicle types in real-time.

## ✨ Features

- **🖼️ Image Detection**: Upload a single image to detect and count vehicles
- **🎬 Video Detection**: Process video files for detection of vehicles in motion
- **📹 Real-time Detection**: Perform live detection and analysis using a camera feed

## 🚀 Detection Capabilities

The system can detect the following vehicle types:
- Cars
- Trucks
- Buses
- Motorcycles
- Bicycles

![Image](https://github.com/user-attachments/assets/2b4dcf9c-b78d-42ec-a59b-06f0e82cfa2b)
![Image](https://github.com/user-attachments/assets/721d37a3-3c8b-45ed-9939-3863674d243d)
![Image](https://github.com/user-attachments/assets/5a8fb20c-7fae-4422-8104-c9d236bb76de)

## 🛠️ Technical Details

- Built with Flask web framework
- Uses YOLOv8x object detection model
- Trained on a comprehensive dataset for high accuracy
- Supports CUDA acceleration for faster processing
- Includes WebM video processing capabilities

## 💻 System Requirements

- Python 3.6+
- PyTorch
- CUDA-enabled GPU (optional but recommended for better performance)
- Flask
- OpenCV

## 📋 Installation

1. Clone this repository
2. Download the model file (vehicle_detection.pt)
3. Install requirements:
   ```bash
   pip install torch flask opencv-python ultralytics
   ```
4. Run the application:
   ```bash
   python app.py
   ```

## 🔧 Usage

After starting the application, navigate to:
- Homepage: `http://localhost:5000/`
- Image Detection: `http://localhost:5000/image`
- Video Detection: `http://localhost:5000/video`
- Real-time Detection: `http://localhost:5000/realtime`

## 🖥️ Web Interface

The project includes a modern and user-friendly web interface with:
- Clean navigation sidebar
- Intuitive upload interfaces
- Real-time processing feedback
- Download capability for processed videos and images

## 🧠 Model Information

The detection model (yolo11n.pt) is a YOLOv11-based model specifically trained to identify various vehicle types with high accuracy in different lighting and environmental conditions.

## 📁 Project Structure

```
├── app.py                   # Main Flask application
├── yolo11n.pt               # Pre-trained YOLO model
├── templates/               # HTML templates for web interface
│   ├── vehicle_home.html            # Homepage template
│   ├── vehicle_image.html           # Image detection page
│   ├── vehicle_video.html           # Video detection page
│   └── vehicle_realtime.html        # Real-time detection page
└── videodetect/             # Output directory for processed videos
```

## 📊 Performance

The system achieves:
- High detection accuracy (95%+)
- Fast processing speeds (30+ FPS with GPU acceleration)
- Reliable performance in various weather and lighting conditions

## 🌐 Applications

This vehicle detection and counting system can be utilized for:
- Urban traffic management
- Smart parking solutions
- Traffic violation detection
- Transportation planning
- Smart city infrastructure

## 📈 Future Enhancements

- Integration with cloud services for remote monitoring
- Mobile application for on-the-go traffic analysis
- Advanced analytics dashboard with historical data
- Multi-camera support for wider area coverage
- Vehicle speed estimation capabilities

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
