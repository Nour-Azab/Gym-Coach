import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import joblib

# Configuration
VIDEO_PATH = 0
MODEL_PATH = r"D:\رواد\Lateral_Raises_code\Gym-Coach-Lateral_Raises\lateral_raises_transformer_autoencoder.pth"
SCALER_PATH = r"D:\رواد\Lateral_Raises_code\Gym-Coach-Lateral_Raises\pose_scaler_lateral_raises.pkl"

WINDOW_SIZE = 30
NUM_FEATURES = 81  # 23 landmarks × 3 (x,y,visibility) +  angles = 69 + 12
ANOMALY_THRESHOLD = 0.815

# Utility Functions
def calculate_angle(a, b, c):
    """Calculate angle in degrees between three points where b is the vertex."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    dot_product = np.dot(ba, bc)
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine_angle = np.clip(dot_product / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))


def extract_pushup_angles_from_landmarks(landmarks_dict):

    # --- Left landmarks ---
    left_shoulder = [landmarks_dict['LEFT_SHOULDER_x'], landmarks_dict['LEFT_SHOULDER_y']]
    left_elbow = [landmarks_dict['LEFT_ELBOW_x'], landmarks_dict['LEFT_ELBOW_y']]
    left_wrist = [landmarks_dict['LEFT_WRIST_x'], landmarks_dict['LEFT_WRIST_y']]
    left_hip = [landmarks_dict['LEFT_HIP_x'], landmarks_dict['LEFT_HIP_y']]
    left_knee = [landmarks_dict['LEFT_KNEE_x'], landmarks_dict['LEFT_KNEE_y']]

    # --- Right landmarks ---
    right_shoulder = [landmarks_dict['RIGHT_SHOULDER_x'], landmarks_dict['RIGHT_SHOULDER_y']]
    right_elbow = [landmarks_dict['RIGHT_ELBOW_x'], landmarks_dict['RIGHT_ELBOW_y']]
    right_wrist = [landmarks_dict['RIGHT_WRIST_x'], landmarks_dict['RIGHT_WRIST_y']]
    right_hip = [landmarks_dict['RIGHT_HIP_x'], landmarks_dict['RIGHT_HIP_y']]
    right_knee = [landmarks_dict['RIGHT_KNEE_x'], landmarks_dict['RIGHT_KNEE_y']]

    # --- Elbow Angles ---
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # --- Shoulder Angles ---
    left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
    right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)

    # --- Hip Angles ---
    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

    # --- Torso Angle ---
    left_hip_vertical = [left_hip[0], left_hip[1] + 1]
    torso_angle = calculate_angle(left_shoulder, left_hip, left_hip_vertical)

    # --- Wrist Alignment Angles ---
    left_wrist_horizontal = [left_wrist[0] + 1, left_wrist[1]]
    right_wrist_horizontal = [right_wrist[0] + 1, right_wrist[1]]

    left_wrist_angle = calculate_angle(left_elbow, left_wrist, left_wrist_horizontal)
    right_wrist_angle = calculate_angle(right_elbow, right_wrist, right_wrist_horizontal)

    # --- Shoulder Elevation ---
    left_shoulder_elevation = left_shoulder[1] - left_hip[1]
    right_shoulder_elevation = right_shoulder[1] - right_hip[1]

    # --- Torso-Hip Drift ---
    torso_hip_drift = left_hip[0] - left_shoulder[0]

    return [
        left_elbow_angle,
        right_elbow_angle,
        left_shoulder_angle,
        right_shoulder_angle,
        left_hip_angle,
        right_hip_angle,
        torso_angle,
        left_wrist_angle,
        right_wrist_angle,
        left_shoulder_elevation,
        right_shoulder_elevation,
        torso_hip_drift
    ]



def draw_angle_visualization(frame, landmarks_dict, h, w):
    """Draw angle measurements on the frame."""
    def draw_angle_text(frame, center, angle, color):
        center_pixel = (int(center[0] * w), int(center[1] * h))
        cv2.putText(frame, f"{int(angle)}", 
                   (center_pixel[0] + 10, center_pixel[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    left_shoulder = [landmarks_dict['LEFT_SHOULDER_x'], landmarks_dict['LEFT_SHOULDER_y']]
    right_shoulder = [landmarks_dict['RIGHT_SHOULDER_x'], landmarks_dict['RIGHT_SHOULDER_y']]
    
    angles = extract_pushup_angles_from_landmarks(landmarks_dict)
    left_shoulder_angle, right_shoulder_angle = angles[0], angles[1]
    
    left_color = (0, 255, 0) if 70 <= left_shoulder_angle <= 100 else (0, 165, 255)
    right_color = (0, 255, 0) if 70 <= right_shoulder_angle <= 100 else (0, 165, 255)
    
    draw_angle_text(frame, left_shoulder, left_shoulder_angle, left_color)
    draw_angle_text(frame, right_shoulder, right_shoulder_angle, right_color)


# Model Definition
class TransformerAutoencoder(nn.Module):
    def __init__(self, num_features, seq_len, d_model=128, nhead=8, num_layers=6):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, num_features)

    def forward(self, x):
        z = self.input_proj(x)
        memory = self.encoder(z)
        reconstructed = self.decoder(z, memory)
        return self.output_proj(reconstructed)


# Keep only pose, hands, and nose
LANDMARK_NAMES = [
    'NOSE',
    'LEFT_SHOULDER', 'RIGHT_SHOULDER',
    'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST',
    'LEFT_PINKY', 'RIGHT_PINKY',
    'LEFT_INDEX', 'RIGHT_INDEX',
    'LEFT_THUMB', 'RIGHT_THUMB',
    'LEFT_HIP', 'RIGHT_HIP',
    'LEFT_KNEE', 'RIGHT_KNEE',
    'LEFT_ANKLE', 'RIGHT_ANKLE',
    'LEFT_HEEL', 'RIGHT_HEEL',
    'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]

LANDMARK_INDICES = {
    'NOSE': 0,
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
    'LEFT_PINKY': 17, 'RIGHT_PINKY': 18,
    'LEFT_INDEX': 19, 'RIGHT_INDEX': 20,
    'LEFT_THUMB': 21, 'RIGHT_THUMB': 22,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
    'LEFT_HEEL': 29, 'RIGHT_HEEL': 30,
    'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32
}


# Initialize
print("=" * 60)
print("Push Up FORM ANALYZER")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

try:
    scaler = joblib.load(SCALER_PATH)
    print(f"Scaler loaded")
except Exception as e:
    print(f"Error loading scaler: {e}")
    exit(1)

try:
    model = TransformerAutoencoder(num_features=NUM_FEATURES, seq_len=WINDOW_SIZE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded ({NUM_FEATURES} features per frame)")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Open Video Source
if isinstance(VIDEO_PATH, int):
    print(f"\nOpening camera at index {VIDEO_PATH}...")
    cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Trying alternative camera indices...")
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                print(f"Opened camera at index {i}")
                VIDEO_PATH = i
                break
        if not cap.isOpened():
            print("Cannot open camera")
            exit(1)
else:
    print(f"\nOpening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video file")
        exit(1)

ret, test_frame = cap.read()
if not ret or test_frame is None:
    cap.release()
    print("Cannot read frames")
    exit(1)

source_type = "Camera" if isinstance(VIDEO_PATH, int) else "Video"
print(f"{source_type} initialized: {test_frame.shape[1]}x{test_frame.shape[0]}")
print(f"Anomaly threshold: {ANOMALY_THRESHOLD:.4f}")
print("\nControls: 'q' = Quit | 'v' = Toggle Angles\n" + "=" * 60)

# Main Loop
pose_data_buffer = []
frame_count = 0
reconstruction_error = 0.0
form_status = "Analyzing..."
status_color = (255, 255, 255)
show_angles = False
pose_detected = False

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read frame")
        break

    frame_count += 1
    h, w, _ = frame.shape
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        pose_detected = True
        
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        landmarks_dict = {}
        for name, idx in LANDMARK_INDICES.items():
            lm = results.pose_landmarks.landmark[idx]
            landmarks_dict[f'{name}_x'] = lm.x
            landmarks_dict[f'{name}_y'] = lm.y
            landmarks_dict[f'{name}_visibility'] = lm.visibility

        if show_angles:
            draw_angle_visualization(frame, landmarks_dict, h, w)

        # Build feature vector
        feature_vector = []
        for name in LANDMARK_NAMES:
            feature_vector.extend([
                landmarks_dict[f'{name}_x'],
                landmarks_dict[f'{name}_y'],
                landmarks_dict[f'{name}_visibility']
            ])

        # Add angles
        angles = extract_pushup_angles_from_landmarks(landmarks_dict)
        feature_vector.extend(angles)

        if len(feature_vector) == NUM_FEATURES:
            pose_data_buffer.append(feature_vector)
        else:
            print(f"Warning: Feature mismatch. Expected {NUM_FEATURES}, got {len(feature_vector)}")
            pose_data_buffer.append([0.0] * NUM_FEATURES)
    else:
        pose_detected = False
        pose_data_buffer.append([0.0] * NUM_FEATURES)

    # Process window
    if len(pose_data_buffer) >= WINDOW_SIZE:
        window = np.array(pose_data_buffer[-WINDOW_SIZE:])
        
        try:
            scaled_window = scaler.transform(window)
            window_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                recon_out = model(window_tensor)
                reconstruction_error = torch.mean((window_tensor - recon_out) ** 2).item()

                if not pose_detected:
                    form_status = "No Pose Detected"
                    status_color = (0, 165, 255)
                elif reconstruction_error > ANOMALY_THRESHOLD:
                    form_status = "POOR FORM!"
                    status_color = (0, 0, 255)
                else:
                    form_status = "Good Form"
                    status_color = (0, 255, 0)
        except Exception as e:
            print(f"Inference error: {e}")
            form_status = "Error"
            status_color = (0, 165, 255)

    # Display UI
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (480, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "LATERAL RAISES ANALYZER", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
    
    cv2.putText(frame, f"Frame: {frame_count} | Buffer: {len(pose_data_buffer)}/{WINDOW_SIZE}", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.putText(frame, f"Reconstruction Error: {reconstruction_error:.6f}", (10, 85),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.putText(frame, f"Threshold: {ANOMALY_THRESHOLD:.4f}", (10, 105),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.putText(frame, f"Form: {form_status}", (10, 135),cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    cv2.putText(frame, "Controls: 'q' = Quit | 'v' = Toggle Angles", (10, h - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if show_angles:
        cv2.putText(frame, "[Angles: ON]", (w - 130, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Lateral Raise Form Analysis", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nQuitting...")
        break
    elif key == ord('v'):
        show_angles = not show_angles
        print(f"Angle visualization: {'ON' if show_angles else 'OFF'}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
pose.close()

print("\n" + "=" * 60)
print("INFERENCE COMPLETE")
print("=" * 60)
print(f"Frames processed: {frame_count}")
print(f"Final reconstruction error: {reconstruction_error:.6f}")
print(f"Anomaly threshold: {ANOMALY_THRESHOLD:.4f}")
print(f"Status: {'POOR FORM' if reconstruction_error > ANOMALY_THRESHOLD else 'Good form'}")
print("=" * 60)