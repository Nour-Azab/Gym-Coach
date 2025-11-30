# ===============================
# PUSH-UPS INFERENCE (FAST + LAZY LOAD)
# ===============================
import cv2
import numpy as np
import traceback

# Lazy global holders
_pose_model = None
_mp_pose = None

# ===============================
# Lazy-load MediaPipe Pose
# ===============================
def get_pose_model():
    global _pose_model, _mp_pose
    if _pose_model is not None:
        return _pose_model

    import mediapipe as mp
    _mp_pose = mp.solutions.pose
    _pose_model = _mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return _pose_model


# ===============================
# 📐 Helper Functions
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
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP',
    'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE'
]

LEFT_LANDMARKS = [
    'LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST',
    'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'
]

RIGHT_LANDMARKS = [
    'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST',
    'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'
]

LANDMARK_INDICES = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
}


# ===============================
# Angle extraction
# ===============================
def extract_angles_from_landmarks(lm):
    left_shoulder = [lm['LEFT_SHOULDER_x'], lm['LEFT_SHOULDER_y']]
    left_elbow = [lm['LEFT_ELBOW_x'], lm['LEFT_ELBOW_y']]
    left_wrist = [lm['LEFT_WRIST_x'], lm['LEFT_WRIST_y']]
    left_hip = [lm['LEFT_HIP_x'], lm['LEFT_HIP_y']]
    left_knee = [lm['LEFT_KNEE_x'], lm['LEFT_KNEE_y']]

    right_shoulder = [lm['RIGHT_SHOULDER_x'], lm['RIGHT_SHOULDER_y']]
    right_elbow = [lm['RIGHT_ELBOW_x'], lm['RIGHT_ELBOW_y']]
    right_wrist = [lm['RIGHT_WRIST_x'], lm['RIGHT_WRIST_y']]
    right_hip = [lm['RIGHT_HIP_x'], lm['RIGHT_HIP_y']]
    right_knee = [lm['RIGHT_KNEE_x'], lm['RIGHT_KNEE_y']]

    # visibility for side detection
    LeftElbow_vis = lm.get("LEFT_ELBOW_visibility", 0)
    RightElbow_vis = lm.get("RIGHT_ELBOW_visibility", 0)
    active_arm = "left" if LeftElbow_vis > RightElbow_vis else "right"

    if active_arm == "left":
        elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
        hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    else:
        elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
        hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

    return [
        active_arm,
        elbow_angle,
        shoulder_angle,
        hip_angle,
    ]


# ===============================
# 🔥 ONNX analyze_frame (fast)
# ===============================
def analyze_frame(frame, session, scaler, threshold, buffer, window_size, num_features, rep_state):
    # Lazy-load pose
    pose = get_pose_model()

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if not results.pose_landmarks:
        buffer.append([0.0] * num_features)
        return {"form_status": "No Pose Detected", "rep_state": rep_state}

    # Landmark dict
    lm = {}
    for name, idx in LANDMARK_INDICES.items():
        p = results.pose_landmarks.landmark[idx]
        lm[f"{name}_x"] = float(p.x)
        lm[f"{name}_y"] = float(p.y)
        lm[f"{name}_visibility"] = float(p.visibility)

    # Extract angles
    angles = extract_angles_from_landmarks(lm)
    active_arm = angles[0]
    elbow_angle = angles[1]

    # Build features: angles + active landmarks
    feature_vector = list(map(float, angles[1:]))

    selection = LEFT_LANDMARKS if active_arm == "left" else RIGHT_LANDMARKS
    for name in selection:
        feature_vector.extend([
            float(lm[f"{name}_x"]),
            float(lm[f"{name}_y"]),
            float(lm[f"{name}_visibility"])
        ])

    if len(feature_vector) == num_features:
        buffer.append(feature_vector)
    else:
        buffer.append([0.0] * num_features)

    # ===============================
    # REP LOGIC (shortened)
    # ===============================
    rep_state.setdefault('prev_angle', None)
    rep_state.setdefault('prev_phase', None)
    rep_state.setdefault('phase', "P1")
    rep_state.setdefault('good_form_flag', False)
    rep_state.setdefault('rep_counter', 0)
    rep_state.setdefault('Top_ROM_error', False)
    rep_state.setdefault('Bottom_ROM_error', False)

    if rep_state['prev_angle'] is None:
        rep_state['prev_angle'] = elbow_angle

    prev_angle = rep_state['prev_angle']
    prev_phase = rep_state['prev_phase']

    # Phases
    if elbow_angle >= 150 and prev_angle < 150:
        rep_state['phase'] = "P1"
        rep_state['Top_ROM_error'] = False
        rep_state['Bottom_ROM_error'] = False
    elif elbow_angle <= 65:
        rep_state['phase'] = "P3"
    elif elbow_angle < prev_angle:
        rep_state['phase'] = "P2"
    elif elbow_angle > prev_angle:
        rep_state['phase'] = "P4"

    # ROM errors
    if prev_phase == "P4" and rep_state["phase"] == "P2":
        rep_state['Top_ROM_error'] = True

    if prev_phase == "P2" and rep_state["phase"] == "P4":
        rep_state['Bottom_ROM_error'] = True

    # Count rep
    if prev_phase == "P4" and rep_state["phase"] == "P1":
        if rep_state["good_form_flag"]:
            rep_state["rep_counter"] += 1
        rep_state["good_form_flag"] = False

    rep_state['prev_phase'] = rep_state['phase']
    rep_state['prev_angle'] = elbow_angle

    # ===============================
    # ONNX inference once window full
    # ===============================
    if len(buffer) >= window_size:
        window = np.array(list(buffer), dtype=np.float32)
        scaled = scaler.transform(window)[np.newaxis, :, :]

        try:
            input_name = session.get_inputs()[0].name
            recon = session.run(None, {input_name: scaled})[0]
        except Exception as e:
            return {
                "form_status": f"ONNX Error: {e}",
                "rep_state": rep_state
            }

        err = float(np.mean((scaled - recon) ** 2))

        if rep_state["Top_ROM_error"]:
            status = "Go up more!"
            rep_state["good_form_flag"] = False
        elif rep_state["Bottom_ROM_error"]:
            status = "Go down more!"
            rep_state["good_form_flag"] = False
        elif err > threshold:
            status = "POOR FORM!"
            rep_state["good_form_flag"] = False
        else:
            status = "Good Form"
            rep_state["good_form_flag"] = True

        return {
            "form_status": status,
            "rep_state": rep_state
        }

    return {"form_status": "Analyzing...", "rep_state": rep_state}


# ===============================
# Reset helper
# ===============================
def reset_rep_counter(rep_state):
    rep_state.update({
        "rep_counter": 0,
        "prev_angle": None,
        "prev_phase": None,
        "phase": "P1",
        "Top_ROM_error": False,
        "Bottom_ROM_error": False,
        "good_form_flag": False,
    })
    return rep_state
