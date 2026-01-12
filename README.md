# Multi-Person Tracking Using YOLOv8 and DeepSORT (OpenCV)

This project implements a robust **multi-person detection and tracking system** using computer vision and deep learning techniques.  
The system processes a video file, detects all people in each frame, assigns a **unique ID to every individual**, and **consistently tracks them across frames**, even during partial occlusions or overlaps.

---

## 🚀 Features

- Accepts a video file as input
- Detects multiple people per frame using **YOLOv8**
- Assigns **unique, persistent IDs** (Person-1, Person-2, etc.)
- Tracks people smoothly across consecutive frames
- Handles **partial occlusions and overlapping individuals**
- Displays real-time bounding boxes and ID labels
- Saves the processed output video

---

## 🧠 System Architecture

Input Video
↓
YOLOv8 (Person Detection)
↓
DeepSORT (Tracking + Re-Identification)
↓
Bounding Boxes + Unique IDs
↓
Output Video


## Why YOLOv8 + DeepSORT?
- **YOLOv8** provides fast and accurate person detection
- **DeepSORT** ensures ID consistency using:
  - Kalman filtering (motion prediction)
  - Appearance-based re-identification (ReID)
  - Data association for overlapping targets

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV**
- **YOLOv8 (Ultralytics)**
- **DeepSORT**
- **NumPy**
- **PyTorch**

---

## 📂 Project Structure

multi-person-tracking-opencv/
│
├── multi_person_tracking.py # Main tracking script
├── README.md # Project documentation
├── requirements.txt # Dependencies
└── .gitignore # Ignored files (videos, venv, etc.)


---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/multi-person-tracking-opencv.git
cd multi-person-tracking-opencv
```

### 2️⃣ Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### ▶️ How to Run

Place a valid video file in the project directory
Rename it to:
```bash
input_video.mp4
```
### 📤 Output

A real-time window displaying:

Bounding boxes around detected people

Unique ID labels (Person-1, Person-2, …)

Output video saved as:
```bash
output_tracked_video.avi
```
## 🎯 Use Cases

Surveillance and security systems

Smart office / building analytics

People counting and movement analysis

Computer vision learning projects

Multi-object tracking demonstrations

## 📌 Notes & Limitations

Input video must not be empty or corrupted

Best performance with:

Fixed camera

Indoor or surveillance-style videos

Extremely dense crowds may reduce tracking accuracy

## 👤 Author

Amaan
Computer Vision & AI Enthusiast
