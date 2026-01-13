import cv2
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# =====================================================
# Global variables
# =====================================================
video_path = None
stop_flag = False

# =====================================================
# Load Models (loaded once)
# =====================================================
model = YOLO("yolov8n.pt")
tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_iou_distance=0.7,
    max_cosine_distance=0.2
)

# =====================================================
# Browse Video Function
# =====================================================
def browse_video():
    global video_path
    video_path = filedialog.askopenfilename(
        title="Select Input Video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )

    if video_path:
        video_label.config(text=os.path.basename(video_path))
    else:
        video_label.config(text="No file selected")

# =====================================================
# Start Tracking Function (Threaded)
# =====================================================
def start_tracking():
    global stop_flag

    if not video_path:
        messagebox.showerror("Error", "Please select a video file first.")
        return

    stop_flag = False
    threading.Thread(target=process_video, daemon=True).start()

# =====================================================
# Stop Tracking Function
# =====================================================
def stop_tracking():
    global stop_flag
    stop_flag = True

# =====================================================
# Main Video Processing Logic
# =====================================================
def process_video():
    global stop_flag

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        "output_tracked_video.avi",
        fourcc,
        fps,
        (width, height)
    )

    while cap.isOpened() and not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO Person Detection
        results = model(frame, conf=0.4, classes=[0])
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])

                bbox = [
                    int(x1),
                    int(y1),
                    int(x2 - x1),
                    int(y2 - y1)
                ]
                detections.append((bbox, conf, "person"))

        # DeepSORT Tracking
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, w, h = map(int, track.to_ltrb())

            cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Person-{track_id}",
                (l, t - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.imshow("Multi-Person Tracking", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# =====================================================
# GUI Setup
# =====================================================
root = tk.Tk()
root.title("Multi-Person Tracking System")
root.geometry("500x250")
root.resizable(False, False)

# Title
title_label = tk.Label(
    root,
    text="Multi-Person Tracking (YOLOv8 + DeepSORT)",
    font=("Arial", 14, "bold")
)
title_label.pack(pady=10)

# Video Selection
frame = tk.Frame(root)
frame.pack(pady=10)

browse_btn = tk.Button(frame, text="Browse Video", width=15, command=browse_video)
browse_btn.grid(row=0, column=0, padx=10)

video_label = tk.Label(frame, text="No file selected", width=30, anchor="w")
video_label.grid(row=0, column=1)

# Control Buttons
control_frame = tk.Frame(root)
control_frame.pack(pady=20)

start_btn = tk.Button(
    control_frame,
    text="Start",
    width=12,
    bg="green",
    fg="white",
    command=start_tracking
)
start_btn.grid(row=0, column=0, padx=15)

stop_btn = tk.Button(
    control_frame,
    text="Stop",
    width=12,
    bg="red",
    fg="white",
    command=stop_tracking
)
stop_btn.grid(row=0, column=1, padx=15)

# Footer
footer = tk.Label(
    root,
    text="Output saved as: output_tracked_video.avi",
    font=("Arial", 9)
)
footer.pack(side="bottom", pady=10)

root.mainloop()
