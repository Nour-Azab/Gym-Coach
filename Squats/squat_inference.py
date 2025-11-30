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

    # Get angles and active arm
    angles = extract_angles_from_landmarks(landmarks_dict)
    active_arm = angles[0]
    knee_angle = angles[1] 

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

    # Rep counting logic based on knee angle
    angle = knee_angle
    
    if angle is not None:
        if rep_state['prev_angle'] is None:
            rep_state['prev_angle'] = angle

        # Phase detection
        if angle >= 160 and rep_state['prev_angle'] < 160:
            rep_state['Bottom_ROM_error'] = False
            rep_state['phase'] = "S1"  # Rest
        elif angle <= 90: 
            rep_state['phase'] = "S3"  # Top (fully contracted)
        elif angle < rep_state['prev_angle'] and angle < 160 and angle > 90: 
            rep_state['phase'] = "S2"  # Going up (contracting)
        elif angle > rep_state['prev_angle'] and angle < 160 and angle > 90: 
            rep_state['phase'] = "S4"  # Going down (extending)

        # Range of Motion Checks
        # 1) Check for incomplete bottom extension
        if rep_state['prev_phase'] is not None:
            if rep_state['phase'] == "S2" and rep_state['prev_phase'] == "S4": 
                rep_state['viable_rep'] = False
                rep_state['Bottom_ROM_error'] = True

        # Rep detection (Going down → Rest)
        if rep_state['prev_phase'] == "S4" and rep_state['phase'] == "S1":
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
            status = "BAD FORM!"
        elif rep_state['Bottom_ROM_error']:
            status = "Not Going Low Enough!"
        elif angle < 160:
            # angles index: 0=active_arm, 1=knee, 2=hip, 3=torso, 4=ankle
            torso_angle = angles[3]
            ankle_angle = angles[4]
            
            if torso_angle > 50:
                 rep_state['viable_rep'] = False
                 status = "Don't Arch Your Back!"
            else:
                 status = "Good Form"
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
    global rep_state
    rep_state = {
        'rep_counter': 0,
        'prev_angle': None,
        'prev_phase': None,
        'phase': "S1",
        'viable_rep': True,
        'Bottom_ROM_error': False
    }