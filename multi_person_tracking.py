import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# =====================================================
# 1. Load YOLOv8 model (Person Detection)
# =====================================================
model = YOLO("yolov8n.pt")  # change to yolov8s.pt for better accuracy

# =====================================================
# 2. Initialize DeepSORT Tracker
# =====================================================
tracker = DeepSort(
    max_age=30,               # keep track alive during short occlusion
    n_init=3,                 # frames before confirming a track
    max_iou_distance=0.7,
    max_cosine_distance=0.2
)

# =====================================================
# 3. Video Input
# =====================================================
VIDEO_PATH = "input_video.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Cannot open input video file.")
    exit()

# =====================================================
# 4. Video Properties (SAFE)
# =====================================================
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 25  # fallback FPS

print(f"Video Properties → Width: {width}, Height: {height}, FPS: {fps}")

# =====================================================
# 5. Output Video Writer (ROBUST)
# =====================================================
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(
    "output_tracked_video.avi",
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    print("ERROR: VideoWriter failed to open.")
    exit()

# =====================================================
# 6. Main Processing Loop
# =====================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------------------------------
    # YOLO Person Detection
    # -------------------------------------------------
    results = model(frame, conf=0.4, classes=[0])  # class 0 = person

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

    # -------------------------------------------------
    # DeepSORT Tracking
    # -------------------------------------------------
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = map(int, track.to_ltrb())

        # Bounding Box
        cv2.rectangle(
            frame,
            (l, t),
            (l + w, t + h),
            (0, 255, 0),
            2
        )

        # ID Label
        cv2.putText(
            frame,
            f"Person-{track_id}",
            (l, t - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # -------------------------------------------------
    # Display & Save
    # -------------------------------------------------
    cv2.imshow("Multi-Person Tracking", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =====================================================
# 7. Cleanup
# =====================================================
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing completed successfully.")
print("Output saved as: output_tracked_video.avi")
