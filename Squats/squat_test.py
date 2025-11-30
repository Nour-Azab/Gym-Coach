import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib
from collections import deque
import cv2
import mediapipe as mp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
VIDEO_PATH = 0
MODEL_PATH = r"best_transformer_autoencoder_squats.pth"
SCALER_PATH = r"pose_scaler_squats.pkl"

WINDOW_SIZE = 30
NUM_FEATURES = 22    #6*3 + 4=22
ANOMALY_THRESHOLD = 0.017

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


def extract_angles_from_landmarks(landmarks_dict):
    # --- Left landmarks ---
    left_shoulder = [landmarks_dict['LEFT_SHOULDER_x'], landmarks_dict['LEFT_SHOULDER_y']]
    left_hip = [landmarks_dict['LEFT_HIP_x'], landmarks_dict['LEFT_HIP_y']]
    left_knee = [landmarks_dict['LEFT_KNEE_x'], landmarks_dict['LEFT_KNEE_y']]
    left_ankle = [landmarks_dict['LEFT_ANKLE_x'], landmarks_dict['LEFT_ANKLE_y']]
    left_foot_index = [landmarks_dict['LEFT_FOOT_INDEX_x'], landmarks_dict['LEFT_FOOT_INDEX_y']]


    # --- Right landmarks ---
    right_shoulder = [landmarks_dict['RIGHT_SHOULDER_x'], landmarks_dict['RIGHT_SHOULDER_y']]
    right_hip = [landmarks_dict['RIGHT_HIP_x'], landmarks_dict['RIGHT_HIP_y']]
    right_knee = [landmarks_dict['RIGHT_KNEE_x'], landmarks_dict['RIGHT_KNEE_y']]
    right_ankle = [landmarks_dict['RIGHT_ANKLE_x'], landmarks_dict['RIGHT_ANKLE_y']]
    right_foot_index = [landmarks_dict['RIGHT_FOOT_INDEX_x'], landmarks_dict['RIGHT_FOOT_INDEX_y']]

  


    # determine active arm
    LeftKnee_vis = landmarks_dict.get("LEFT_KNEE_visibility", 0)
    RightKnee_vis = landmarks_dict.get("RIGHT_KNEE_visibility", 0)
    active_arm = "left" if LeftKnee_vis > RightKnee_vis else "right"

    # vertical reference points 
    vertical_point_left = [left_hip[0], left_hip[1] - 1]
    vertical_point_right = [right_hip[0], right_hip[1] - 1]

  


    if active_arm == "left":
        knee_angles=calculate_angle(left_hip, left_knee, left_ankle) # knee tmm
        hip_angles=calculate_angle(left_shoulder, left_hip, left_knee)
        torso_angles=calculate_angle(left_shoulder, left_hip, vertical_point_left)
        ankle_angles=calculate_angle(left_knee, left_ankle, left_foot_index)

    else:
        knee_angles=calculate_angle(right_hip, right_knee, right_ankle) # knee tmm
        hip_angles=calculate_angle(right_shoulder, right_hip, right_knee)
        torso_angles=calculate_angle(right_shoulder, right_hip, vertical_point_right)
        ankle_angles=calculate_angle(right_knee, right_ankle, right_foot_index)
        

  

    return [
       active_arm,
       knee_angles,
       hip_angles,
       torso_angles,
       ankle_angles
     
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
    
    angles = extract_angles_from_landmarks(landmarks_dict)
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


# MediaPipe Landmark Mapping (Upper Body Only)
LANDMARK_NAMES = [
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 
    'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
    'LEFT_ANKLE', 'RIGHT_ANKLE','LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX',
    'LEFT_HEEL', 'RIGHT_HEEL'

]

LEFT_LANDMARKS  = [
 'LEFT_SHOULDER',   'LEFT_HIP', 'LEFT_KNEE','LEFT_ANKLE','LEFT_FOOT_INDEX','LEFT_HEEL'
 ]

RIGHT_LANDMARKS =  ['RIGHT_SHOULDER',   'RIGHT_HIP', 'RIGHT_KNEE','RIGHT_ANKLE','RIGHT_FOOT_INDEX','RIGHT_HEEL']



LANDMARK_INDICES = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12, 
    'LEFT_HIP': 23, 'RIGHT_HIP': 24, 'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32,
    'LEFT_HEEL': 29, 'RIGHT_HEEL': 30
}


# Initialize
print("=" * 60)
print("LATERAL RAISES FORM ANALYZER (UPPER BODY)")
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
status_color = (255, 255, 255)  # White for initial state

print("Starting inference... Press 'q' to quit")

rep_counter = 0
prev_angle = None
prev_phase = None
phase = "S1"
viable_rep = True
Bottom_ROM_error = False

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

        

        # Build feature vector (14 landmarks × 3 = 42)
        feature_vector = []
        angles = extract_angles_from_landmarks(landmarks_dict)
        activee_arm= angles[0]
           # Add 5 angles
        feature_vector.extend(angles[1:])

        if activee_arm =='left':
            selection= LEFT_LANDMARKS
        else :
            selection= RIGHT_LANDMARKS

        for name in selection:

            feature_vector.extend([
           
                landmarks_dict[f'{name}_x'],
                landmarks_dict[f'{name}_y'],
                landmarks_dict[f'{name}_visibility'],
                
            ])

     
        Anamoly_Threshold = 0.032
        angle=angles[1]
        
        if angle is not None:
                if prev_angle is None:
                    prev_angle = angle

                if angle > 160 and prev_angle <= 160:
                    Bottom_ROM_error = False
                    phase = "S1"  # stand
                elif angle <= 90:
                    phase = "S3"  # bottom
                elif angle < prev_angle and angle <= 160 and angle > 90:
                    phase = "S2"  # going down
                elif angle > prev_angle and angle <= 160 and angle > 90:
                    phase = "S4"  # going up

                #Range of Motion Checks
                # Check for not deep enough rep
                if prev_angle is not None:
                    if phase == "S2" and prev_phase == "S4":
                        viable_rep = False
                        Bottom_ROM_error = True
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
                form_status = "Bad Form!"
                status_color = (0, 0, 255)  # Red
            elif Bottom_ROM_error:
                form_status = "Not Going Low Enough!"
                status_color = (0, 0, 255) 
            elif angle < 160:
                if angles[3] > 50:
                    viable_rep = False
                    form_status = "Don't Arch Your Back!"
                    status_color = (0, 0, 255) 
                else:
                    form_status = "Normal Form!"
                    status_color = (0, 0, 255)
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