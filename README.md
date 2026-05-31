# Motion-Tracker-V1
A modern desktop motion detection and object tracking application built with Python, OpenCV, and PyQt6.

Motion Tracker V2 provides real-time motion detection, object tracking, object classification, automatic recording, live analytics, and support for webcams, video files, and IP camera streams.

**Features**

**Real-Time Motion Detection**

* Detects movement using background subtraction.
* Tracks multiple moving objects simultaneously.
* Adjustable detection sensitivity.
* Configurable minimum object size filtering.

**Object Tracking**

* Persistent object IDs.
* Motion trails for tracked objects.
* Object timeout management.
* Multiple visualization modes.

 **Object Classification**

Objects are automatically classified as:

* Person
* Vehicle
* Animal
* Unknown

Uses OpenCV HOG-based human detection combined with additional classification heuristics.

**Visualization Modes**

* Box Mode
* Dot Mode
* Outline Mode
* Heatmap Mode

**Motion Recording**

Automatically records video clips when motion is detected.

Features:

* Motion-triggered recording
* Automatic clip saving
* Adjustable recording timeout
* Custom recording directory

**Alert System**

Receive notifications when activity is detected.

* Sound alerts
* Windows toast notifications
* Configurable alert cooldown

**Live Dashboard**

Monitor application performance in real time.

* FPS graph
* Active object count graph
* Object classification breakdown
* Saved recordings log

**Input Sources**

Supports:

* Webcam devices
* Video files
* RTSP streams
* HTTP video streams
* IP cameras

---

**Screenshots**

Add screenshots here after deployment.

Example:

```text
tracker_shot_001.png
tracker_shot_004.png
tracker_shot_006.png
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/motion-tracker-v2.git
cd motion-tracker-v2
```

### Install Dependencies

```bash
pip install opencv-python PyQt6 numpy
```

Optional Windows notifications:

```bash
pip install win10toast
```

---

## Running

```bash
python motion_tracker_v2.py
```

---

## Building Executable

Create a standalone desktop executable:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed motion_tracker_v2.py
```

---

## Usage

### Start Tracking

1. Select a camera.
2. Enter an IP camera URL or RTSP stream if required.
3. Click **START**.

### Recording Motion

1. Enable **Auto-record motion clips**.
2. Select a save directory.
3. Motion clips will automatically be recorded when movement is detected.

### Switching Display Modes

Choose between:

* BOX
* DOT
* OUTLINE
* HEATMAP

from the Display panel.

---

## Technologies Used

* Python
* OpenCV
* PyQt6
* NumPy

---

## Project Structure

```text
motion-tracker-v2/
│
├── motion_tracker_v2.py
├── recordings/
├── screenshots/
├── README.md
└── requirements.txt
```

---

## Future Improvements

* YOLO-based object detection
* GPU acceleration
* Web dashboard
* Facial recognition support
* Cloud recording storage
* Mobile notifications
* Cross-platform notification support

---

## Requirements

* Python 3.10+
* OpenCV
* PyQt6
* NumPy

Recommended:

* Webcam or IP camera
* Windows 10/11 for native notifications

---

## License

MIT License

---

## Author

Developed as a computer vision and surveillance analytics project using Python and OpenCV.
