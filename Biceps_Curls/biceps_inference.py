import cv2
import torch
import numpy as np
import mediapipe as mp
import joblib
import torch.nn as nn

# ===============================
# Model Definition
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
# Helper Functions
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

LEFT_LANDMARKS  = [
 'LEFT_SHOULDER',  'LEFT_ELBOW', 'LEFT_WRIST',  'LEFT_PINKY', 'LEFT_INDEX', 'LEFT_THUMB','LEFT_HIP', 
 ]

RIGHT_LANDMARKS =  ['RIGHT_SHOULDER',  'RIGHT_ELBOW','RIGHT_WRIST',  'RIGHT_PINKY','RIGHT_INDEX',  'RIGHT_THUMB','RIGHT_HIP']



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


    # determine active arm
    LeftElbow_vis = landmarks_dict.get("LEFT_ELBOW_visibility", 0)
    RightElbow_vis = landmarks_dict.get("RIGHT_ELBOW_visibility", 0)
    active_arm = "left" if LeftElbow_vis > RightElbow_vis else "right"

    # vertical reference points 
    vertical_point_left  = [left_shoulder[0],  left_shoulder[1]  - 1]
    vertical_point_right = [right_shoulder[0], right_shoulder[1] - 1]


    if active_arm == "left":
        elbow_fexion_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        torso_lean_angle = calculate_angle(left_hip, left_shoulder, vertical_point_left) if left_hip[0] else 0.0
        upper_arm_torso_angle = calculate_angle(left_hip, left_shoulder, left_elbow) if left_hip[0] else 0.0
        left_index = [left_wrist[0] + (left_wrist[0] - left_elbow[0]),
                          left_wrist[1] + (left_wrist[1] - left_elbow[1])]
        wrist_angle =calculate_angle(left_elbow, left_wrist, left_index)
        forearm_vertical_angle=calculate_angle(vertical_point_left, left_elbow, left_wrist)

    else:
        elbow_fexion_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        torso_lean_angle = calculate_angle(right_hip, right_shoulder, vertical_point_right) if right_hip[0] else 0.0
        upper_arm_torso_angle = calculate_angle(right_hip, right_shoulder, right_elbow) if right_hip[0] else 0.0
        right_index = [right_wrist[0] + (right_wrist[0] - right_elbow[0]),
                          right_wrist[1] + (right_wrist[1] - right_elbow[1])]
        wrist_angle =calculate_angle(right_elbow, right_wrist, right_index)
        forearm_vertical_angle=calculate_angle(vertical_point_right, right_elbow, right_wrist)

    return [
       active_arm,
     elbow_fexion_angle,
    forearm_vertical_angle,
    wrist_angle,
    upper_arm_torso_angle,
    torso_lean_angle
    ]


# ===============================
# Inference Function
# ===============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def analyze_frame(frame, model, scaler, threshold, device, buffer, window_size, num_features,rep_state):

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        buffer.append([0.0] * num_features)
        return {
            "form_status": "No Person Detected", 
            "rep_state": rep_state
        }

    # Extract landmarks
    landmarks_dict = {}
    for name, idx in LANDMARK_INDICES.items():
        lm = results.pose_landmarks.landmark[idx]
        landmarks_dict[f"{name}_x"] = lm.x
        landmarks_dict[f"{name}_y"] = lm.y
        landmarks_dict[f"{name}_visibility"] = lm.visibility

    # Get angles and active arm
    angles = extract_angles_from_landmarks(landmarks_dict)
    active_arm = angles[0]
    elbow_flexion_angle = angles[1]

    # Build feature vector starting with angles (excluding active_arm string)
    feature_vector = []
    feature_vector.extend(angles[1:])  # Add 5 angles

    # Select landmarks based on active arm
    if active_arm == 'left':
        selection = LEFT_LANDMARKS
    else:
        selection = RIGHT_LANDMARKS

    # Add landmark coordinates for active side
    for name in selection:
        feature_vector.extend([
            landmarks_dict[f'{name}_x'],
            landmarks_dict[f'{name}_y'],
            landmarks_dict[f'{name}_visibility']
        ])

    # Append to buffer
    if len(feature_vector) == num_features:
        buffer.append(feature_vector)
    else:
        print(f"Warning: Feature mismatch. Expected {num_features}, got {len(feature_vector)}")
        buffer.append([0.0] * num_features)

    # Rep counting logic based on elbow flexion angle
    angle = elbow_flexion_angle
    
    if angle is not None:
        if rep_state['prev_angle'] is None:
            rep_state['prev_angle'] = angle

        # Phase detection
        if angle >= 160 and rep_state['prev_angle'] < 160:
            rep_state['Top_ROM_error'] = False
            rep_state['Bottom_ROM_error'] = False
            rep_state['phase'] = "B1"  # Rest
        elif angle <= 60:
            rep_state['phase'] = "B3"  # Top (fully contracted)
        elif angle < rep_state['prev_angle'] and angle < 160 and angle > 60:
            rep_state['phase'] = "B2"  # Going up (contracting)
        elif angle > rep_state['prev_angle'] and angle < 160 and angle > 60:
            rep_state['phase'] = "B4"  # Going down (extending)

        # Range of Motion Checks
        # 1) Check for incomplete bottom extension
        if rep_state['prev_phase'] is not None:
            if rep_state['phase'] == "B2" and rep_state['prev_phase'] == "B4":
                rep_state['viable_rep'] = False
                rep_state['Bottom_ROM_error'] = True

        # 2) Check for weak contraction at the top
        if rep_state['phase'] == "B4" and rep_state['prev_phase'] == "B2":
            rep_state['viable_rep'] = False
            rep_state['Top_ROM_error'] = True

        # Rep detection (Going down → Rest)
        if rep_state['prev_phase'] == "B4" and rep_state['phase'] == "B1":
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

        # Determine form status based on reconstruction error and ROM errors
        if err > threshold:
            rep_state['viable_rep'] = False
            status = "POOR FORM!"
        elif rep_state['Bottom_ROM_error']:
            status = "Extend your arms more!"
        elif rep_state['Top_ROM_error']:
            status = "Contract your arms more!"
        else:
            status = "Good Form"

        return {
            "form_status": status, 
            "reconstruction_error": err,
            "rep_state": rep_state
        }

    return {
        "form_status": "Analyzing...", 
        "rep_state": rep_state
    }


def reset_rep_counter():
    global rep_state
    rep_state = {
        'rep_counter': 0,
        'prev_angle': None,
        'prev_phase': None,
        'phase': "B1",
        'viable_rep': True,
        'Top_ROM_error': False,
        'Bottom_ROM_error': False
    }