import cv2
from ultralytics import YOLO
from tracker import Tracker

# --- Load model ---
model = YOLO("models/yolo12s.pt")  # Ensure this path exists

# --- Load video ---
video_path = "videos/road_feed_org.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Error: Could not open video at {video_path}")
    exit()

# --- Initialize tracker ---
tracker = Tracker()

# --- Get video properties ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video loaded: {frame_width}x{frame_height} @ {fps:.2f} FPS")

# --- Vehicle class mapping ---
vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
vehicle_counts = {"up": {v: 0 for v in vehicle_classes.values()},
                  "down": {v: 0 for v in vehicle_classes.values()}}

# --- Line setup ---
line_position = frame_height // 2

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or read error.")
        break

    # Convert BGR to RGB for YOLO
    results = model(frame[..., ::-1])

    detections = []
    for r in results:
        for box in r.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box[:6]
            class_id = int(class_id)
            if class_id in vehicle_classes:
                detections.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(score), class_id])

    tracked_objects = tracker.update(detections)

    for obj in tracked_objects:
        x, y, w, h, score, class_id, obj_id = obj
        class_name = vehicle_classes[class_id]
        center_y = y + h // 2
        direction = "down" if center_y > line_position else "up"

        # Draw box
        color = (0, 255, 0) if direction == "down" else (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{class_name}-{obj_id}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Count logic (only count once when crossing line)
        if direction == "down" and center_y > line_position - 5 and center_y < line_position + 5:
            vehicle_counts["down"][class_name] += 1
        elif direction == "up" and center_y > line_position - 5 and center_y < line_position + 5:
            vehicle_counts["up"][class_name] += 1

    # Draw line
    cv2.line(frame, (0, line_position), (frame_width, line_position), (0, 0, 255), 2)

    # Display counts
    y_pos = 30
    for dir in vehicle_counts:
        for cls, count in vehicle_counts[dir].items():
            cv2.putText(frame, f"{cls} {dir}: {count}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_pos += 25

    cv2.imshow("Vehicle Detection and Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
