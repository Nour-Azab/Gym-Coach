# ===============================
# SQUAT INFERENCE (FAST + LAZY LOAD)
# ===============================
import cv2
import numpy as np
import traceback

# Lazy global mediapipe instance
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
# 📐 Angle Calculation
# ===============================
def calculate_angle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    dot = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cos_val = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


# ===============================
# Landmark configuration
# ===============================
LEFT_LANDMARKS = [
    "LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE",
    "LEFT_ANKLE", "LEFT_FOOT_INDEX", "LEFT_HEEL"
]

RIGHT_LANDMARKS = [
    "RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE",
    "RIGHT_ANKLE", "RIGHT_FOOT_INDEX", "RIGHT_HEEL"
]

LANDMARK_INDICES = {
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29, "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32
}


# ===============================
# Angle Extraction
# ===============================
def extract_angles_from_landmarks(lm):
    left_shoulder = [lm["LEFT_SHOULDER_x"], lm["LEFT_SHOULDER_y"]]
    left_hip = [lm["LEFT_HIP_x"], lm["LEFT_HIP_y"]]
    left_knee = [lm["LEFT_KNEE_x"], lm["LEFT_KNEE_y"]]
    left_ankle = [lm["LEFT_ANKLE_x"], lm["LEFT_ANKLE_y"]]
    left_foot = [lm["LEFT_FOOT_INDEX_x"], lm["LEFT_FOOT_INDEX_y"]]

    right_shoulder = [lm["RIGHT_SHOULDER_x"], lm["RIGHT_SHOULDER_y"]]
    right_hip = [lm["RIGHT_HIP_x"], lm["RIGHT_HIP_y"]]
    right_knee = [lm["RIGHT_KNEE_x"], lm["RIGHT_KNEE_y"]]
    right_ankle = [lm["RIGHT_ANKLE_x"], lm["RIGHT_ANKLE_y"]]
    right_foot = [lm["RIGHT_FOOT_INDEX_x"], lm["RIGHT_FOOT_INDEX_y"]]

    left_vis = lm.get("LEFT_KNEE_visibility", 0)
    right_vis = lm.get("RIGHT_KNEE_visibility", 0)
    active = "left" if left_vis > right_vis else "right"

    if active == "left":
        knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        torso_angle = calculate_angle(left_shoulder, left_hip, [left_hip[0], left_hip[1] - 1.0])
        ankle_angle = calculate_angle(left_knee, left_ankle, left_foot)
    else:
        knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        torso_angle = calculate_angle(right_shoulder, right_hip, [right_hip[0], right_hip[1] - 1.0])
        ankle_angle = calculate_angle(right_knee, right_ankle, right_foot)

    return [
        active,
        knee_angle,
        hip_angle,
        torso_angle,
        ankle_angle
    ]


# ===============================
# analyze_frame (FAST ONNX)
# ===============================
def analyze_frame(frame, session, scaler, threshold, buffer, window_size, num_features, rep_state):

    pose = get_pose_model()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if not results.pose_landmarks:
        buffer.append([0.0] * num_features)
        return {"form_status": "No Pose Detected", "rep_state": rep_state}

    lm = {}
    for name, idx in LANDMARK_INDICES.items():
        p = results.pose_landmarks.landmark[idx]
        lm[f"{name}_x"] = float(p.x)
        lm[f"{name}_y"] = float(p.y)
        lm[f"{name}_visibility"] = float(p.visibility)

    angles = extract_angles_from_landmarks(lm)
    active = angles[0]
    knee_angle = angles[1]

    feature_vector = [float(x) for x in angles[1:]]

    selection = LEFT_LANDMARKS if active == "left" else RIGHT_LANDMARKS
    for name in selection:
        feature_vector.extend([
            lm.get(f"{name}_x", 0),
            lm.get(f"{name}_y", 0),
            lm.get(f"{name}_visibility", 0)
        ])

    buffer.append(feature_vector if len(feature_vector) == num_features else [0]*num_features)

    # =========================
    # REP LOGIC (optimized same behavior)
    # =========================
    rep_state.setdefault("prev_angle", None)
    rep_state.setdefault("prev_phase", None)
    rep_state.setdefault("phase", "S1")
    rep_state.setdefault("Bottom_ROM_error", False)
    rep_state.setdefault("good_form_flag", False)
    rep_state.setdefault("rep_counter", 0)

    if rep_state["prev_angle"] is None:
        rep_state["prev_angle"] = knee_angle

    prev_angle = rep_state["prev_angle"]
    prev_phase = rep_state["prev_phase"]

    if knee_angle >= 160 and prev_angle < 160:
        rep_state["Bottom_ROM_error"] = False
        rep_state["phase"] = "S1"
    elif knee_angle <= 90:
        rep_state["phase"] = "S3"
    elif knee_angle < prev_angle:
        rep_state["phase"] = "S2"
    elif knee_angle > prev_angle:
        rep_state["phase"] = "S4"

    if prev_phase == "S4" and rep_state["phase"] == "S2":
        rep_state["Bottom_ROM_error"] = True

    if prev_phase == "S4" and rep_state["phase"] == "S1":
        if rep_state["good_form_flag"]:
            rep_state["rep_counter"] += 1
        rep_state["good_form_flag"] = False

    rep_state["prev_phase"] = rep_state["phase"]
    rep_state["prev_angle"] = knee_angle

    # =========================
    # ONNX inference
    # =========================
    if len(buffer) >= window_size:
        window = np.array(list(buffer), dtype=np.float32)
        scaled = scaler.transform(window)[np.newaxis, :, :]

        input_name = session.get_inputs()[0].name
        recon = session.run(None, {input_name: scaled})[0]

        err = float(np.mean((scaled - recon) ** 2))

        if err > threshold:
            status = "BAD FORM!"
            rep_state["good_form_flag"] = False
        elif rep_state["Bottom_ROM_error"]:
            status = "Not Going Low Enough!"
            rep_state["good_form_flag"] = False
        elif angles[3] > 50:
            status = "Don't Arch Your Back!"
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
# Reset template
# ===============================
def reset_rep_counter():
    return {
        "rep_counter": 0,
        "prev_angle": None,
        "prev_phase": None,
        "phase": "S1",
        "Bottom_ROM_error": False,
        "good_form_flag": False
    }
