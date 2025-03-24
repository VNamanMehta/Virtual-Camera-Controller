# Virtual Camera Controller

This project is a **Virtual Camera Controller** that uses **YOLOv8** for real-time object detection and segmentation. It allows users to apply effects like background blur, black background, or custom backgrounds to their webcam feed and stream the processed video to a virtual camera.

## Features
- **Real-time Object Detection and Segmentation** using YOLOv8.
- **Background Effects**:
  - Blur the background.
  - Replace the background with a custom image.
  - Black out the background.
- **Virtual Camera Integration**: Streams the processed video to a virtual camera for use in applications like gmeet etc.
- **Web Interface**: A simple web-based UI to control the camera, FPS, blur strength, and background settings.

## Requirements
- **Python 3.8+**
- **OBS Studio**: Ensure OBS Studio is installed and running in the background when using this application. OBS is required to utilize the virtual camera feature.

## Installation
1. Clone this repository:
   ```bash
   https://github.com/VNamanMehta/Virtual-Camera-Controller.git
   cd virtual-camera-controller
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure you have OBS Studio installed and running in the background.

## Usage
1. Start the application:
   ```
   python main.py
   ```
   
2. Open OBS Studio and ensure that it is running in the background.

3. Access the web interface:
   Open your browser and navigate to ```http://localhost:8000```.

4. Use the web interface to:
   Select a camera.
   Adjust FPS and blur strength.
   Choose a background effect.
   The processed video will be streamed to the virtual camera. 
