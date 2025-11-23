import cv2
import torch
import numpy as np
import mediapipe as mp
import joblib
import torch.nn as nn

# ===============================
# ⚙️ Model Definition
# ===============================
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


# ===============================
# 📐 Helper Functions
# ===============================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    dot = np.dot(ba, bc)
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

# MediaPipe Landmark Mapping (Upper Body Only)
LANDMARK_NAMES = [
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
    'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
    'LEFT_HIP', 'RIGHT_HIP'
]

LANDMARK_INDICES = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12, 'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16, 'LEFT_PINKY': 17, 'RIGHT_PINKY': 18,
    'LEFT_INDEX': 19, 'RIGHT_INDEX': 20, 'LEFT_THUMB': 21, 'RIGHT_THUMB': 22,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24
}

def extract_angles_from_landmarks(landmarks_dict):
    # --- Left landmarks ---
    left_shoulder = [landmarks_dict['LEFT_SHOULDER_x'], landmarks_dict['LEFT_SHOULDER_y']]
    left_elbow = [landmarks_dict['LEFT_ELBOW_x'], landmarks_dict['LEFT_ELBOW_y']]
    left_wrist = [landmarks_dict['LEFT_WRIST_x'], landmarks_dict['LEFT_WRIST_y']]
    left_hip = [landmarks_dict['LEFT_HIP_x'], landmarks_dict['LEFT_HIP_y']]

    # --- Right landmarks ---
    right_shoulder = [landmarks_dict['RIGHT_SHOULDER_x'], landmarks_dict['RIGHT_SHOULDER_y']]
    right_elbow = [landmarks_dict['RIGHT_ELBOW_x'], landmarks_dict['RIGHT_ELBOW_y']]
    right_wrist = [landmarks_dict['RIGHT_WRIST_x'], landmarks_dict['RIGHT_WRIST_y']]
    right_hip = [landmarks_dict['RIGHT_HIP_x'], landmarks_dict['RIGHT_HIP_y']]

    # --- Shoulder Angles (Arm Abduction) ---
    left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
    right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)

    # --- Elbow Angles (Flexion/Extension) ---
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # --- Torso Lean Angles ---
    left_hip_vertical = [left_hip[0], left_hip[1] + 1]
    right_hip_vertical = [right_hip[0], right_hip[1] + 1]

    left_torso_angle = calculate_angle(left_shoulder, left_hip, left_hip_vertical)
    right_torso_angle = calculate_angle(right_shoulder, right_hip, right_hip_vertical)

    # --- Shoulder Elevation (shrugging detection) ---
    left_shoulder_elevation = left_shoulder[1] - left_hip[1]
    right_shoulder_elevation = right_shoulder[1] - right_hip[1]

    # --- Wrist Alignment Angles ---
    left_wrist_horizontal = [left_wrist[0] + 1, left_wrist[1]]
    right_wrist_horizontal = [right_wrist[0] + 1, right_wrist[1]]

    left_wrist_angle = calculate_angle(left_elbow, left_wrist, left_wrist_horizontal)
    right_wrist_angle = calculate_angle(right_elbow, right_wrist, right_wrist_horizontal)

    # --- Arm Drift (forward/backward) ---
    left_arm_drift = left_elbow[0] - left_shoulder[0]
    right_arm_drift = right_elbow[0] - right_shoulder[0]

    return [
        left_shoulder_angle,
        right_shoulder_angle,
        left_elbow_angle,
        right_elbow_angle,
        left_torso_angle,
        right_torso_angle,
        left_shoulder_elevation,
        right_shoulder_elevation,
        left_wrist_angle,
        right_wrist_angle,
        left_arm_drift,
        right_arm_drift
    ]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def analyze_frame(frame, model, scaler, threshold, device, buffer, window_size, num_features,rep_state):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        buffer.append([0.0] * num_features)
        return {
            "form_status": "No Pose Detected", 
            "rep_state": rep_state
        }

    # Extract landmarks
    landmarks_dict = {}
    for name, idx in LANDMARK_INDICES.items():
        lm = results.pose_landmarks.landmark[idx]
        landmarks_dict[f"{name}_x"] = lm.x
        landmarks_dict[f"{name}_y"] = lm.y
        landmarks_dict[f"{name}_visibility"] = lm.visibility

    # Build feature vector (14 landmarks × 3 = 42)
    feature_vector = []
    for name in LANDMARK_NAMES:
        feature_vector.extend([
            landmarks_dict[f'{name}_x'],
            landmarks_dict[f'{name}_y'],
            landmarks_dict[f'{name}_visibility']
        ])

    # Get angles and add to feature vector
    angles = extract_angles_from_landmarks(landmarks_dict)
    feature_vector.extend(angles)

    # Calculate average shoulder angle for rep counting
    angle = (angles[0] + angles[1]) / 2  # Average of left and right shoulder angles

    # Append to buffer
    if len(feature_vector) == num_features:
        buffer.append(feature_vector)
    else:
        print(f"Warning: Feature mismatch. Expected {num_features}, got {len(feature_vector)}")
        buffer.append([0.0] * num_features)

    # Rep counting logic based on shoulder angle
    if angle is not None:
        if rep_state['prev_angle'] is None:
            rep_state['prev_angle'] = angle

        # Phase detection for lateral raises
        if angle <= 30 and rep_state['prev_angle'] > 30:
            rep_state['Top_ROM_error'] = False
            rep_state['Bottom_ROM_error'] = False
            rep_state['phase'] = "LR1"  # Rest position (arms down)
        elif angle >= 75:
            rep_state['phase'] = "LR3"  # Top position (arms raised)
        elif angle > rep_state['prev_angle'] and 30 < angle < 75:
            rep_state['phase'] = "LR2"  # Going up (raising arms)
        elif angle < rep_state['prev_angle'] and 30 < angle < 75:
            rep_state['phase'] = "LR4"  # Going down (lowering arms)

        # Range of Motion Checks
        # 1) Check for incomplete range at top (didn't raise high enough)
        if rep_state['prev_phase'] is not None:
            if rep_state['phase'] == "LR4" and rep_state['prev_phase'] == "LR2":
                rep_state['viable_rep'] = False
                rep_state['Top_ROM_error'] = True

        # 2) Check for incomplete range at bottom (didn't lower enough)
        if rep_state['phase'] == "LR2" and rep_state['prev_phase'] == "LR4":
            rep_state['viable_rep'] = False
            rep_state['Bottom_ROM_error'] = True

        # Rep detection (Going down → Rest)
        if rep_state['prev_phase'] == "LR4" and rep_state['phase'] == "LR1":
            if rep_state['viable_rep']:
                rep_state['rep_counter'] += 1
            rep_state['viable_rep'] = True

        rep_state['prev_phase'] = rep_state['phase']
        rep_state['prev_angle'] = angle

    # Only analyze when buffer is full
    if len(buffer) >= window_size:
        window = np.array(list(buffer), dtype=np.float32)

        try:
            scaled = scaler.transform(window)
        except Exception as e:
            return {
                "form_status": f"Scaler Error: {e}", 
                "rep_state": rep_state
            }

        tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            recon = model(tensor)
            err = torch.mean((tensor - recon) ** 2).item()

        # Determine form status based on reconstruction error and specific form checks
        if err > threshold:
            rep_state['viable_rep'] = False
            status = "POOR FORM!"
        elif rep_state['Top_ROM_error']:
            status = "Raise elbows higher!"
        elif rep_state['Bottom_ROM_error']:
            status = "Relax arms at the end!"
        # Check if wrist is higher than elbow during the raise
        elif angle > 45:
            if (landmarks_dict['RIGHT_WRIST_y'] < landmarks_dict['RIGHT_ELBOW_y'] or 
                landmarks_dict['LEFT_WRIST_y'] < landmarks_dict['LEFT_ELBOW_y']):
                rep_state['viable_rep'] = False
                status = "Wrist higher than elbow!"
        else:
            status = "Good Form"

        return {
            "form_status": status, 
            "rep_state": rep_state
        }

    return {
        "form_status": "Analyzing...", 
        "rep_state": rep_state
    }


def reset_rep_counter():
    """Call this function to reset the rep counter (e.g., at start of new set)"""
    global rep_state
    rep_state = {
        'rep_counter': 0,
        'prev_angle': None,
        'prev_phase': None,
        'phase': "LR1",
        'viable_rep': True,
        'Top_ROM_error': False,
        'Bottom_ROM_error': False
    }