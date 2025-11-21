import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

# Configuration
VIDEO_PATH = 0  # Use 0 for webcam or provide video path
MODEL_PATH = r"squat_transformer_autoencoder.pth"
SCALER_PATH = r"pose_scaler.pkl"

WINDOW_SIZE = 30
NUM_FEATURES = 44  # 12 landmarks × 3 (x,y,visibility) + 8 angles = 36 + 8 = 44
Anamoly_Threshold = 0.06  # Adjust based on your trained model's threshold


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


def extract_angles_from_landmarks(landmarks_dict):
    """Calculate pose angles from landmark coordinates."""
    angles = []
    
    # Left side angles
    left_knee_angle = calculate_angle(
        [landmarks_dict['LEFT_HIP_x'], landmarks_dict['LEFT_HIP_y']],
        [landmarks_dict['LEFT_KNEE_x'], landmarks_dict['LEFT_KNEE_y']],
        [landmarks_dict['LEFT_ANKLE_x'], landmarks_dict['LEFT_ANKLE_y']]
    )
    
    right_knee_angle = calculate_angle(
        [landmarks_dict['RIGHT_HIP_x'], landmarks_dict['RIGHT_HIP_y']],
        [landmarks_dict['RIGHT_KNEE_x'], landmarks_dict['RIGHT_KNEE_y']],
        [landmarks_dict['RIGHT_ANKLE_x'], landmarks_dict['RIGHT_ANKLE_y']]
    )
    
    left_hip_angle = calculate_angle(
        [landmarks_dict['LEFT_SHOULDER_x'], landmarks_dict['LEFT_SHOULDER_y']],
        [landmarks_dict['LEFT_HIP_x'], landmarks_dict['LEFT_HIP_y']],
        [landmarks_dict['LEFT_KNEE_x'], landmarks_dict['LEFT_KNEE_y']]
    )
    
    right_hip_angle = calculate_angle(
        [landmarks_dict['RIGHT_SHOULDER_x'], landmarks_dict['RIGHT_SHOULDER_y']],
        [landmarks_dict['RIGHT_HIP_x'], landmarks_dict['RIGHT_HIP_y']],
        [landmarks_dict['RIGHT_KNEE_x'], landmarks_dict['RIGHT_KNEE_y']]
    )
    
    # Torso angles (using vertical reference)
    left_hip_vertical = [landmarks_dict['LEFT_HIP_x'], landmarks_dict['LEFT_HIP_y'] - 1]
    left_torso_angle = calculate_angle(
        [landmarks_dict['LEFT_SHOULDER_x'], landmarks_dict['LEFT_SHOULDER_y']],
        [landmarks_dict['LEFT_HIP_x'], landmarks_dict['LEFT_HIP_y']],
        left_hip_vertical
    )
    
    right_hip_vertical = [landmarks_dict['RIGHT_HIP_x'], landmarks_dict['RIGHT_HIP_y'] - 1]
    right_torso_angle = calculate_angle(
        [landmarks_dict['RIGHT_SHOULDER_x'], landmarks_dict['RIGHT_SHOULDER_y']],
        [landmarks_dict['RIGHT_HIP_x'], landmarks_dict['RIGHT_HIP_y']],
        right_hip_vertical
    )
    
    # Ankle angles
    left_ankle_angle = calculate_angle(
        [landmarks_dict['LEFT_KNEE_x'], landmarks_dict['LEFT_KNEE_y']],
        [landmarks_dict['LEFT_ANKLE_x'], landmarks_dict['LEFT_ANKLE_y']],
        [landmarks_dict['LEFT_FOOT_INDEX_x'], landmarks_dict['LEFT_FOOT_INDEX_y']]
    )
    
    right_ankle_angle = calculate_angle(
        [landmarks_dict['RIGHT_KNEE_x'], landmarks_dict['RIGHT_KNEE_y']],
        [landmarks_dict['RIGHT_ANKLE_x'], landmarks_dict['RIGHT_ANKLE_y']],
        [landmarks_dict['RIGHT_FOOT_INDEX_x'], landmarks_dict['RIGHT_FOOT_INDEX_y']]
    )
    
    return [left_knee_angle, right_knee_angle, left_hip_angle, right_hip_angle,
            left_torso_angle, right_torso_angle, left_ankle_angle, right_ankle_angle]


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
        recon_out = self.output_proj(reconstructed)
        return recon_out


# MediaPipe landmark indices mapping
LANDMARK_NAMES = [
    'LEFT_SHOULDER', 'RIGHT_SHOULDER',
    'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
    'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
    'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]

LANDMARK_INDICES = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24, 'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28, 'LEFT_HEEL': 29, 'RIGHT_HEEL': 30,
    'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32
}


# Load model and scaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = joblib.load(SCALER_PATH)

model = TransformerAutoencoder(num_features=NUM_FEATURES, seq_len=WINDOW_SIZE).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model and scaler loaded successfully")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open video source (camera or video file)
if isinstance(VIDEO_PATH, int):
    # Camera mode - use DirectShow backend for Windows
    print(f"Opening camera at index {VIDEO_PATH}...")
    cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera at index {VIDEO_PATH}")
        print("Trying different camera indices...")
        # Try other camera indices
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                print(f"Successfully opened camera at index {i}")
                VIDEO_PATH = i
                break
        if not cap.isOpened():
            raise IOError(f"Cannot open any camera. Please check your camera connection.")
else:
    # Video file mode
    print(f"Opening video file: {VIDEO_PATH}...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {VIDEO_PATH}. Please check the file path.")

# Test if video source is working
ret, test_frame = cap.read()
if not ret or test_frame is None:
    cap.release()
    raise IOError("Video source opened but cannot read frames.")
    
source_type = "Camera" if isinstance(VIDEO_PATH, int) else "Video"
print(f"{source_type} initialized successfully. Resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")

# Buffers
pose_data_buffer = []
frame_count = 0
reconstruction_error = 0.0
form_status = "Analyzing..."
status_color = (255, 255, 255)  # White for initial state

print("Starting inference... Press 'q' to quit")

rep_counter = 0
prev_angle = None
prev_phase = None
phase = "S1"
while True:
    success, frame = cap.read()
    if not success or frame is None:
        print("Warning: Failed to read frame")
        break

    frame_count += 1
    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        # Extract landmarks (excluding face landmarks as in training)
        landmarks_dict = {}
        for name, idx in LANDMARK_INDICES.items():
            lm = results.pose_landmarks.landmark[idx]
            landmarks_dict[f'{name}_x'] = lm.x
            landmarks_dict[f'{name}_y'] = lm.y
            landmarks_dict[f'{name}_visibility'] = lm.visibility

        # Build feature vector (66 landmark features)
        feature_vector = []
        for name in LANDMARK_NAMES:
            feature_vector.extend([
                landmarks_dict[f'{name}_x'],
                landmarks_dict[f'{name}_y'],
                landmarks_dict[f'{name}_visibility']
            ])

        # Calculate angles (8 angle features)
        angles = extract_angles_from_landmarks(landmarks_dict)
        feature_vector.extend(angles)

        if landmarks_dict['LEFT_KNEE_visibility'] > landmarks_dict['RIGHT_KNEE_visibility']:
            angle = angles[0]
            Anamoly_Threshold = 0.065
        else:
            angle = angles[1]
            Anamoly_Threshold = 0.045

        
        if angle is not None:
                if prev_angle is None:
                    prev_angle = angle

                if angle > 160 and prev_angle <= 160:
                    phase = "S1"  # stand
                elif angle <= 90:
                    phase = "S3"  # bottom
                elif angle < prev_angle and angle <= 160 and angle > 90:
                    phase = "S2"  # going down
                elif angle > prev_angle and angle <= 160 and angle > 90:
                    phase = "S4"  # going up
                # Rep detection (bottom → stand)
                if prev_phase == "S4" and phase == "S1":
                    if viable_rep:
                        rep_counter += 1
                        print(f"Rep completed! Total reps: {rep_counter}")
                    viable_rep = True

                prev_phase = phase
                prev_angle = angle

        # Total: 36 + 8 = 44 features (adjust if your NUM_FEATURES is different)
        if len(feature_vector) == NUM_FEATURES:
            pose_data_buffer.append(feature_vector)
        else:
            print(f"Warning: Feature mismatch. Expected {NUM_FEATURES}, got {len(feature_vector)}")
            pose_data_buffer.append([0.0] * NUM_FEATURES)
    else:
        pose_data_buffer.append([0.0] * NUM_FEATURES)

    # Process window when buffer is full
    if len(pose_data_buffer) >= WINDOW_SIZE:
        window = np.array(pose_data_buffer[-WINDOW_SIZE:])
        scaled_window = scaler.transform(window)
        window_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            recon_out = model(window_tensor)
            reconstruction_error = torch.mean((window_tensor - recon_out) ** 2).item()
            
            # Determine form status
            if reconstruction_error > Anamoly_Threshold:
                viable_rep = False
                form_status = "ANOMALY DETECTED!"
                status_color = (0, 0, 255)  # Red
            else:
                form_status = "Normal Form"
                status_color = (0, 255, 0)  # Green

    # Display information
    cv2.rectangle(frame, (4, 4), (400, 100), (0, 0, 0), -1)
    
    cv2.putText(frame, f"Frame: {frame_count}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Recon Error: {reconstruction_error:.6f}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Status: {form_status}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    cv2.putText(frame, f"Rep Counter: {rep_counter}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Squat Form Analysis - Autoencoder", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()

print(f"\nInference complete. Processed {frame_count} frames.")
print(f"Final reconstruction error: {reconstruction_error:.6f}")