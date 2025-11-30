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
    'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
    'LEFT_ANKLE', 'RIGHT_ANKLE','LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX',
    'LEFT_HEEL', 'RIGHT_HEEL'

]

LEFT_LANDMARKS  = [
 'LEFT_SHOULDER',  'LEFT_ELBOW', 'LEFT_WRIST',  'LEFT_PINKY', 'LEFT_INDEX', 'LEFT_THUMB','LEFT_HIP', 'LEFT_KNEE','LEFT_ANKLE','LEFT_FOOT','LEFT_HEEL'
 ]

RIGHT_LANDMARKS =  ['RIGHT_SHOULDER',  'RIGHT_ELBOW','RIGHT_WRIST',  'RIGHT_PINKY','RIGHT_INDEX',  'RIGHT_THUMB','RIGHT_HIP', 'RIGHT_KNEE','RIGHT_ANKLE','RIGHT_FOOT','RIGHT_HEEL']



LANDMARK_INDICES = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12, 'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16, 'LEFT_PINKY': 17, 'RIGHT_PINKY': 18,
    'LEFT_INDEX': 19, 'RIGHT_INDEX': 20, 'LEFT_THUMB': 21, 'RIGHT_THUMB': 22,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24, 'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32,
    'LEFT_HEEL': 29, 'RIGHT_HEEL': 30
}

def extract_angles_from_landmarks(landmarks_dict):
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


    # determine active arm
    LeftElbow_vis = landmarks_dict.get("LEFT_ELBOW_visibility", 0)
    RightElbow_vis = landmarks_dict.get("RIGHT_ELBOW_visibility", 0)
    active_arm = "left" if LeftElbow_vis > RightElbow_vis else "right"

    # vertical reference points 
    vertical_point_left = [left_hip[0], left_hip[1] + 1]
    vertical_point_right = [right_hip[0], right_hip[1] + 1]

    # horizontal reference points 
    horizontal_point_left = [left_wrist[0]+1, left_wrist[1]]
    horizontal_point_right = [right_wrist[0]+1, right_wrist[1]]


    if active_arm == "left":
        elbow_flexion_angle=calculate_angle(left_shoulder, left_elbow, left_wrist) # elbow tmm
        shoulder_angle=calculate_angle(left_elbow, left_shoulder, left_hip)
        hip_angle=(calculate_angle(left_shoulder, left_hip, left_knee))
        torso_angle=(calculate_angle(left_shoulder, left_hip, vertical_point_left))
        wrist_angle=(calculate_angle(left_elbow, left_wrist, horizontal_point_left))
        shoulder_elev=(left_shoulder[1]-left_hip[1])
        torso_hip_drift=(left_hip[0]-left_shoulder[0])

    else:
        elbow_flexion_angle=calculate_angle(right_shoulder, right_elbow, right_wrist) # elbow tmm
        shoulder_angle=calculate_angle(right_elbow, right_shoulder, right_hip)
        hip_angle=(calculate_angle(right_shoulder, right_hip, right_knee))
        torso_angle=(calculate_angle(right_shoulder, right_hip, vertical_point_right))
        wrist_angle=(calculate_angle(right_elbow, right_wrist, horizontal_point_right))
        shoulder_elev=(right_shoulder[1]-right_hip[1])
        torso_hip_drift=(right_hip[0]-right_shoulder[0])
        
    return [
       active_arm,
       elbow_flexion_angle,
       shoulder_angle,
       hip_angle,
       torso_angle,
       wrist_angle,
       shoulder_elev,
       torso_hip_drift
     
    ]

# ===============================
# Inference Function
# ===============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def analyze_frame(frame, model, scaler, threshold, device, buffer, window_size, num_features, rep_state):
   
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
    elbow_flexion_angle = angles[1]

    # Build feature vector starting with angles (excluding active_arm string)
    feature_vector = []
    feature_vector.extend(angles[1:])  # Add 7 angles

    # Select landmarks based on active arm
    if active_arm == 'left':
        selection = LEFT_LANDMARKS
    else:
        selection = RIGHT_LANDMARKS

    # Add landmark coordinates for active side (11 landmarks × 3 = 33)
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

        # Phase detection for push-ups
        if angle >= 150 and rep_state['prev_angle'] < 150:
            rep_state['Top_ROM_error'] = False
            rep_state['Bottom_ROM_error'] = False
            rep_state['phase'] = "P1"  # Up position (arms extended)
        elif angle <= 65:
            rep_state['phase'] = "P3"  # Bottom position (arms bent)
        elif angle < rep_state['prev_angle'] and angle < 150 and angle > 65:
            rep_state['phase'] = "P2"  # Going down (lowering)
        elif angle > rep_state['prev_angle'] and angle < 150 and angle > 65:
            rep_state['phase'] = "P4"  # Going up (pushing)

        # Range of Motion Checks
        # 1) Check for incomplete bottom range (didn't go down enough)
        if rep_state['prev_phase'] is not None:
            if rep_state['phase'] == "P2" and rep_state['prev_phase'] == "P4":
                rep_state['viable_rep'] = False
                rep_state['Top_ROM_error'] = True

        # 2) Check for incomplete top range (didn't extend fully)
        if rep_state['phase'] == "P4" and rep_state['prev_phase'] == "P2":
            rep_state['viable_rep'] = False
            rep_state['Bottom_ROM_error'] = True

        # Rep detection (Going up → Top position)
        if rep_state['prev_phase'] == "P4" and rep_state['phase'] == "P1":
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
        if rep_state['Top_ROM_error']:
            status = "Go up more!"
        elif rep_state['Bottom_ROM_error']:
            status = "Go down more!"
        elif err > threshold:
            rep_state['viable_rep'] = False
            status = "POOR FORM!"
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

def reset_rep_counter(rep_state):
    """Reset the rep counter in the given rep_state dictionary"""
    rep_state['rep_counter'] = 0
    rep_state['prev_angle'] = None
    rep_state['prev_phase'] = None
    rep_state['phase'] = "P1"
    rep_state['viable_rep'] = True
    rep_state['Top_ROM_error'] = False
    rep_state['Bottom_ROM_error'] = False
    return rep_state