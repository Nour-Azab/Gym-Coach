import cv2
import mediapipe as mp
import os
import csv
import pandas as pd

# ------------------ Setup ------------------
VIDEO_PATH = r"video.mp4" # video path
OUTPUT_FOLDER = r"" # frames output if you want to save
CSV_FILE = r"pose_landmarks_temp.csv" # location to save csv file of the video

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {VIDEO_PATH}")

# ------------------ Prepare CSV ------------------
landmark_names = [lm.name for lm in mp_pose.PoseLandmark]
header = ["frame", "video_name"]
for name in landmark_names:
    header += [f"{name}_x", f"{name}_y", f"{name}_visibility"]

csv_data = []

# ------------------ Pose Estimation ------------------
video_name = os.path.basename(VIDEO_PATH)
frame_count = 0

print("=" * 60)
print("POSE ESTIMATION - VIDEO PROCESSING")
print("=" * 60)
print(f"Video: {video_name}")
print(f"Output folder: {OUTPUT_FOLDER}")
print(f"CSV file: {CSV_FILE}")
print("=" * 60)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while True:
        success, frame = cap.read()
        if not success:
            print("\nEnd of video or failed to read frame.")
            break

        frame_count += 1
        h, w, _ = frame.shape

        # Convert BGR → RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        row = [frame_count, video_name]

        if results.pose_landmarks:
            # Draw skeleton on frame
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )

            # Extract landmark coordinates
            for lm in results.pose_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                vis = lm.visibility
                row += [cx, cy, vis]
        else:
            # No pose detected - fill with empty values
            row += [""] * (len(header) - len(row))

        csv_data.append(row)

        # ---------------- Display Info ----------------
        cv2.putText(frame, f"Frame: {frame_count}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        if results.pose_landmarks:
            cv2.putText(frame, "Pose: DETECTED", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Pose: NOT DETECTED", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Save frame
        # output_path = os.path.join(OUTPUT_FOLDER, f"frame_{frame_count:04d}.jpg")
        # cv2.imwrite(output_path, frame)

        # Resize for display
        display_frame = cv2.resize(frame, (800, 600))
        cv2.imshow("Pose Estimation", display_frame)

        # Progress indicator
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nUser interrupted processing.")
            break

# ------------------ Save CSV ------------------
print("\n" + "=" * 60)
print("Saving data to CSV...")

with open(CSV_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(csv_data)

df = pd.DataFrame(csv_data, columns=header)

print(f"CSV saved: {CSV_FILE}")
print(f"Total frames processed: {frame_count}")
print(f"Frames saved to: {OUTPUT_FOLDER}")
print(f"DataFrame shape: {df.shape}")
print("=" * 60)

# Display summary
print("\nData Summary:")
print(df.head())
print("\nLandmark columns:", len([col for col in df.columns if '_x' in col]))

cap.release()
cv2.destroyAllWindows()