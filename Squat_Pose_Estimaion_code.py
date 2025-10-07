import cv2
import mediapipe as mp
import os
import csv
import pandas as pd
import math

# ------------------ Setup ------------------
VIDEO_PATH = r"C:\Users\Abdallah\Downloads\gettyimages-1476795398-640_adpp.mp4"
OUTPUT_FOLDER = r"D:\رواد\frames"
CSV_FILE = r"D:\رواد\pose_landmarks_.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {VIDEO_PATH}")

# ------------------ Prepare CSV ------------------
landmark_names = [lm.name for lm in mp_pose.PoseLandmark]
header = ["frame", "video_name", "rep_counter"]
for name in landmark_names:
    header += [f"{name}_x", f"{name}_y", f"{name}_visibility"]
header.append("phase")

csv_data = []

# ------------------ Helper: Compute angle ------------------
def compute_angle_with_vertical(hip, knee):
    """Compute angle between vertical (upward) and thigh vector (hip->knee)."""
    thigh_vec = (knee[0] - hip[0], knee[1] - hip[1])
    vertical_vec = (0, -1)
    dot = thigh_vec[0] * vertical_vec[0] + thigh_vec[1] * vertical_vec[1]
    mag_thigh = math.sqrt(thigh_vec[0] ** 2 + thigh_vec[1] ** 2)
    if mag_thigh == 0:
        return None

    cos_theta = dot / mag_thigh
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    angle = math.degrees(math.acos(cos_theta))

    if angle > 90:
        angle = 180 - angle
    return angle

# ------------------ Pose Estimation ------------------
rep_counter = 0
video_name = os.path.basename(VIDEO_PATH)
frame_count = 0
prev_angle = None
prev_phase = None
phase = "S1"

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while True:
        success, frame = cap.read()
        if not success:
            print("End of video or failed to read frame.")
            break

        frame_count += 1

        # Automatically handle any video size (no distortion)
        h, w, _ = frame.shape

        # Convert BGR → RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        row = [frame_count, video_name, rep_counter]
        angle = None

        if results.pose_landmarks:
            # Draw skeleton on original frame size
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )

            coords = []
            for lm in results.pose_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                vis = lm.visibility
                coords.append((cx, cy, vis))
                row += [cx, cy, vis]

            # Choose leg with higher visibility
            left_hip = coords[mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = coords[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_hip = coords[mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_knee = coords[mp_pose.PoseLandmark.RIGHT_KNEE.value]

            left_vis = left_hip[2] + left_knee[2]
            right_vis = right_hip[2] + right_knee[2]
            angle = compute_angle_with_vertical(
                right_hip, right_knee
            ) if right_vis > left_vis else compute_angle_with_vertical(
                left_hip, left_knee
            )

            # ---------------- Phases (renamed only) ----------------
            if angle is not None:
                if prev_angle is None:
                    prev_angle = angle

                # Going down
                if angle > 35 and prev_angle <= 35:
                    phase = "S2"  # previously "down"

                # Reaching bottom
                elif angle >= 70:
                    phase = "S3"  # previously "bottom"

                # Going up
                elif angle < 50 and prev_angle >= 50:
                    phase = "S4"  # previously "up"

                # Fully standing
                elif angle <= 25:
                    phase = "S1"  # previously "stand"

                # Rep detection (bottom → stand)
                if prev_phase == "S4" and phase == "S1":
                    rep_counter += 1
                    print(f"✅ Rep completed! Total reps: {rep_counter}")

                prev_phase = phase
                prev_angle = angle
        else:
            row += [""] * (len(header) - len(row) - 1)

        # Append phase
        row.append(phase)
        csv_data.append(row)

        # ---------------- Display Info ----------------
        cv2.putText(frame, f"Frame: {frame_count}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Phase: {phase}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
        cv2.putText(frame, f"Reps: {rep_counter}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 3)

        # Save frame (keeps original resolution)
        output_path = os.path.join(OUTPUT_FOLDER, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_path, frame)

        # Resize only for display (not for processing)
        display_frame = cv2.resize(frame, (800, 600))
        cv2.imshow("Pose Estimation", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ------------------ Save CSV ------------------
with open(CSV_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(csv_data)

df = pd.DataFrame(csv_data, columns=header)
print("\nFinal Landmark Data with Phase:")
print(df['phase'].value_counts())

cap.release()
cv2.destroyAllWindows()
