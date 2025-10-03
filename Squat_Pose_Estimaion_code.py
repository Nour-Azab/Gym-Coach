import cv2
import mediapipe as mp
import os
import csv
import pandas as pd
import math

# ------------------ Setup ------------------
VIDEO_PATH = r"C:\Users\Youss\OneDrive\Documents\DEPI\Final Project\all_squat_videos\6dddd176-2460-48cf-bd01-3eaaa0dc754f.mp4"

OUTPUT_FOLDER = "pose_frames"
CSV_FILE = "pose_landmarks.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)

# ------------------ Prepare CSV ------------------
landmark_names = [lm.name for lm in mp_pose.PoseLandmark]

# (x,y,visibility)
header = ["frame"]
header.append('video_name')
header.append('rep_counter')

for name in landmark_names:
    header.append(f"{name}_x")
    header.append(f"{name}_y")
    header.append(f"{name}_visibility")
header.append("phase")

csv_data = []

# ------------------ Helper: Compute angle ------------------
def compute_angle_with_vertical(hip, knee):
    """
    Compute angle between vertical (upward) and thigh vector (hip->knee).
    hip, knee = (x, y, visibility)
    """
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
video_name = "vid_0"
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

    frame_count = 0
    prev_angle = None
    prev_phase = None
    phase = "s1"

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("End of video or failed to read frame.")
            break

        frame = cv2.resize(frame, (800, 600))
        frame_count += 1

        # Convert BGR → RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        row = [frame_count]
        row.append(video_name)
        row.append(rep_counter)

        if results.pose_landmarks:
            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )

            h, w, _ = frame.shape
            coords = []

            for lm in results.pose_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                vis = lm.visibility
                coords.append((cx, cy, vis))

         
                row.append(cx)
                row.append(cy)
                row.append(vis)

            # Extract hip and knee
            left_hip = coords[mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = coords[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_hip = coords[mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_knee = coords[mp_pose.PoseLandmark.RIGHT_KNEE.value]

            # Choose leg based on visibility
            left_vis = left_hip[2] + left_knee[2]
            right_vis = right_hip[2] + right_knee[2]

            if right_vis > left_vis:
                angle = compute_angle_with_vertical(right_hip, right_knee)
                print(f"Frame {frame_count}: Using RIGHT leg")
            else:
                angle = compute_angle_with_vertical(left_hip, left_knee)
                print(f"Frame {frame_count}: Using LEFT leg")

            print(f"  Hip-Knee Angle with Vertical: {angle}")

            if angle is not None:
                if prev_angle is None: # to initialize only the prev_angle for the first time
                    prev_angle = angle
                if angle <= 32:
                    if phase == "s4" and angle < prev_angle:   # checking if person is going up
                        phase = "s5"
                    elif phase == "s5" and angle > prev_angle: # checking if person is going down
                        phase = "s1"
                elif 32 < angle <= 75:
                    if phase == 's1':
                        phase = "s2"
                    elif phase == 's3':
                        phase = "s4"
                elif 75 < angle <= 120 and phase == 's2':
                    phase = "s3"

            if prev_phase != phase:  # only check on phase change
                if prev_phase == "s5" and phase == "s1":
                    rep_counter += 1
                    print(f"Rep completed! Total reps: {rep_counter}")

            prev_phase = phase
            prev_angle = angle


        else:
            row.extend([""] * (len(header) - len(row)))  # fill until full length
            row[-1] = phase  # last column is phase

        # Append phase
        row.append(phase)
        csv_data.append(row)

        # Show frame number + phase
        cv2.putText(frame, f"Frame {frame_count} | Phase: {phase}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Save frame
        output_path = os.path.join(OUTPUT_FOLDER, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_path, frame)

        cv2.imshow("Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# ------------------ Save CSV ------------------
with open(CSV_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(csv_data)

df = pd.DataFrame(csv_data, columns=header)
print("\nFinal Landmark Data with Phase:")
print(df)



cap.release()
cv2.destroyAllWindows()