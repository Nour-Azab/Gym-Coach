# ===============================
# FAST & LAZY INFERENCE VERSION (Lateral Raises)
# ===============================
import cv2
import numpy as np
import traceback

# NOTE: we intentionally DO NOT import mediapipe or onnxruntime at module import time
# They are heavy; we import them lazily inside get_pose_model() or when needed.

_pose_model = None
_mp_pose = None

# ===============================
# Lazy-load MediaPipe Pose
# ===============================
def get_pose_model():
    global _pose_model, _mp_pose
    if _pose_model is not None:
        return _pose_model

    # Lazy import (heavy)
    import mediapipe as mp
    _mp_pose = mp.solutions.pose
    _pose_model = _mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return _pose_model


# ===============================
# 📐 Angle Calculation
# ===============================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    dot = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


# ===============================
# Landmark Mapping
# ===============================
LANDMARK_NAMES = [
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
    'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
    'LEFT_HIP', 'RIGHT_HIP'
]

LANDMARK_INDICES = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
    'LEFT_PINKY': 17, 'RIGHT_PINKY': 18,
    'LEFT_INDEX': 19, 'RIGHT_INDEX': 20,
    'LEFT_THUMB': 21, 'RIGHT_THUMB': 22,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24
}


# ===============================
# Angle Extraction (unchanged logic)
# ===============================
def extract_angles_from_landmarks(lm):
    left_shoulder = [lm['LEFT_SHOULDER_x'], lm['LEFT_SHOULDER_y']]
    left_elbow = [lm['LEFT_ELBOW_x'], lm['LEFT_ELBOW_y']]
    left_wrist = [lm['LEFT_WRIST_x'], lm['LEFT_WRIST_y']]
    left_hip = [lm['LEFT_HIP_x'], lm['LEFT_HIP_y']]

    right_shoulder = [lm['RIGHT_SHOULDER_x'], lm['RIGHT_SHOULDER_y']]
    right_elbow = [lm['RIGHT_ELBOW_x'], lm['RIGHT_ELBOW_y']]
    right_wrist = [lm['RIGHT_WRIST_x'], lm['RIGHT_WRIST_y']]
    right_hip = [lm['RIGHT_HIP_x'], lm['RIGHT_HIP_y']]

    # Shoulder Angles
    left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
    right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)

    # Elbow Angles
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Torso Lean
    left_hip_vertical = [left_hip[0], left_hip[1] + 1]
    right_hip_vertical = [right_hip[0], right_hip[1] + 1]

    left_torso_angle = calculate_angle(left_shoulder, left_hip, left_hip_vertical)
    right_torso_angle = calculate_angle(right_shoulder, right_hip, right_hip_vertical)

    # Shoulder Elevation
    left_shoulder_elev = left_shoulder[1] - left_hip[1]
    right_shoulder_elev = right_shoulder[1] - right_hip[1]

    # Wrist angle
    left_wrist_horizontal = [left_wrist[0] + 1, left_wrist[1]]
    right_wrist_horizontal = [right_wrist[0] + 1, right_wrist[1]]

    left_wrist_angle = calculate_angle(left_elbow, left_wrist, left_wrist_horizontal)
    right_wrist_angle = calculate_angle(right_elbow, right_wrist, right_wrist_horizontal)

    # Arm drift
    left_arm_drift = left_elbow[0] - left_shoulder[0]
    right_arm_drift = right_elbow[0] - right_shoulder[0]

    return [
        left_shoulder_angle, right_shoulder_angle,
        left_elbow_angle, right_elbow_angle,
        left_torso_angle, right_torso_angle,
        left_shoulder_elev, right_shoulder_elev,
        left_wrist_angle, right_wrist_angle,
        left_arm_drift, right_arm_drift
    ]


# ===============================
# ONNX Analyze Frame (lazy pose + same logic)
# ===============================
def analyze_frame(frame, session, scaler, threshold, buffer, window_size, num_features, rep_state):
    try:
        pose = get_pose_model()  # lazy-load mediapipe pose
    except Exception as e:
        # If mediapipe fails, return quickly
        return {"form_status": f"Pose Load Error: {e}", "rep_state": rep_state}

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        buffer.append([0.0] * num_features)
        return {"form_status": "No Pose Detected", "rep_state": rep_state}

    # Extract landmarks
    lm = {}
    for name, idx in LANDMARK_INDICES.items():
        p = results.pose_landmarks.landmark[idx]
        lm[f"{name}_x"] = p.x
        lm[f"{name}_y"] = p.y
        lm[f"{name}_visibility"] = p.visibility

    # Build feature vector: landmarks then angles
    feature_vector = []
    for name in LANDMARK_NAMES:
        feature_vector.extend([lm[f"{name}_x"], lm[f"{name}_y"], lm[f"{name}_visibility"]])

    # Extract angles and extend
    angles = extract_angles_from_landmarks(lm)
    feature_vector.extend(angles)

    # Average shoulder angle for rep logic
    angle = (angles[0] + angles[1]) / 2.0 if angles else 0.0

    # Buffer
    if len(feature_vector) == num_features:
        buffer.append(feature_vector)
    else:
        buffer.append([0.0] * num_features)

    # ===============================
    # REP STATE DEFAULTS
    # ===============================
    rep_state.setdefault('prev_angle', None)
    rep_state.setdefault('prev_phase', None)
    rep_state.setdefault('phase', "LR1")
    rep_state.setdefault('viable_rep', True)
    rep_state.setdefault('Top_ROM_error', False)
    rep_state.setdefault('Bottom_ROM_error', False)
    rep_state.setdefault('good_form_flag', False)
    rep_state.setdefault('rep_counter', 0)

    # ===============================
    # REP LOGIC (unchanged)
    # ===============================
    if rep_state['prev_angle'] is None:
        rep_state['prev_angle'] = angle

    prev_angle = rep_state['prev_angle']
    prev_phase = rep_state['prev_phase']

    if angle <= 30 and prev_angle > 30:
        rep_state['Top_ROM_error'] = False
        rep_state['Bottom_ROM_error'] = False
        rep_state['phase'] = "LR1"
    elif angle >= 75:
        rep_state['phase'] = "LR3"
    elif angle > prev_angle and 30 < angle < 75:
        rep_state['phase'] = "LR2"
    elif angle < prev_angle and 30 < angle < 75:
        rep_state['phase'] = "LR4"

    if prev_phase is not None:
        if rep_state['phase'] == "LR4" and prev_phase == "LR2":
            rep_state['Top_ROM_error'] = True
            rep_state['viable_rep'] = False

    if rep_state['phase'] == "LR2" and prev_phase == "LR4":
        rep_state['Bottom_ROM_error'] = True
        rep_state['viable_rep'] = False

    if prev_phase == "LR4" and rep_state["phase"] == "LR1":
        if rep_state["good_form_flag"]:
            rep_state["rep_counter"] += 1
        rep_state["good_form_flag"] = False
        rep_state["viable_rep"] = True

    rep_state['prev_phase'] = rep_state['phase']
    rep_state['prev_angle'] = angle

    # ===============================
    # ONNX INFERENCE
    # ===============================
    if len(buffer) >= window_size:
        window = np.array(list(buffer), dtype=np.float32)

        try:
            scaled = scaler.transform(window)
        except Exception as e:
            return {"form_status": f"Scaler Error: {e}", "rep_state": rep_state}

        onnx_input = scaled[np.newaxis, :, :].astype(np.float32)

        try:
            input_name = session.get_inputs()[0].name
            recon = session.run(None, {input_name: onnx_input})[0]
        except Exception as e:
            tb = traceback.format_exc()
            return {"form_status": f"ONNX Run Error: {e}", "trace": tb, "rep_state": rep_state}

        err = float(np.mean((onnx_input - recon) ** 2))

        if err > threshold:
            status = "POOR FORM!"
            rep_state["good_form_flag"] = False

        elif rep_state["Top_ROM_error"]:
            status = "Raise elbows higher!"
            rep_state["good_form_flag"] = False

        elif rep_state["Bottom_ROM_error"]:
            status = "Relax arms at the end!"
            rep_state["good_form_flag"] = False

        elif angle > 45:
            if (lm.get('RIGHT_WRIST_y', 1) < lm.get('RIGHT_ELBOW_y', 1) or
                lm.get('LEFT_WRIST_y', 1) < lm.get('LEFT_ELBOW_y', 1)):
                status = "Wrist higher than elbow!"
                rep_state["good_form_flag"] = False
            else:
                status = "Good Form"
                rep_state["good_form_flag"] = True
        else:
            status = "Good Form"
            rep_state["good_form_flag"] = True

        return {
            "form_status": status,
            "rep_state": rep_state
        }

    return {"form_status": "Analyzing...", "rep_state": rep_state}


# ===============================
# Reset rep counter
# ===============================
def reset_rep_counter():
    return {
        'rep_counter': 0,
        'prev_angle': None,
        'prev_phase': None,
        'phase': "LR1",
        'viable_rep': True,
        'Top_ROM_error': False,
        'Bottom_ROM_error': False,
        'good_form_flag': False
    }
