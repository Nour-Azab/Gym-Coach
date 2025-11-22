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

def calculate_angle(a, b, c):
    """
    Calculates the angle in degrees between three points (a, b, c),
    where 'b' is the vertex.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    dot_product = np.dot(ba, bc)
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine_angle = np.clip(dot_product / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

class TransformerAutoencoder(nn.Module):
    def __init__(self, num_features, seq_len, d_model=128, nhead=8, num_layers=6):
        super().__init__()
        self.seq_len = seq_len

        # Encoder
        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, num_features)

    def forward(self, x):
        z = self.input_proj(x)
        memory = self.encoder(z)
        reconstructed = self.decoder(z, memory)
        recon_out = self.output_proj(reconstructed)
        return recon_out

# --- 2. NEW FUNCTION FOR LIVE LANDMARK PROCESSING ---


def process_landmarks_to_angles(landmarks, img_width, img_height):


    # Helper to get landmark coordinates and visibility
    def get_landmark(name):
        try:
            lm = landmarks.landmark[mp_pose.PoseLandmark[name].value]
            # Check visibility
            if lm.visibility < 0.4:
                return None
            return [lm.x * img_width, lm.y * img_height]
        except:
            return None

    # Get key landmarks
    left_shoulder = get_landmark('LEFT_SHOULDER')
    left_elbow = get_landmark('LEFT_ELBOW')
    left_wrist = get_landmark('LEFT_WRIST')
    left_hip = get_landmark('LEFT_HIP')

    right_shoulder = get_landmark('RIGHT_SHOULDER')
    right_elbow = get_landmark('RIGHT_ELBOW')
    right_wrist = get_landmark('RIGHT_WRIST')
    right_hip = get_landmark('RIGHT_HIP')
    left_vis = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility
    right_vis = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility

    active_arm = "left" if left_vis > right_vis else "right"

    # Vertical reference points
    vertical_point_left  = [left_shoulder[0],  left_shoulder[1]  - 100] if left_shoulder else None
    vertical_point_right = [right_shoulder[0], right_shoulder[1] - 100] if right_shoulder else None

    # Compute angles for active arm only
    try:
        if active_arm == "left":
            if not all([left_shoulder, left_elbow, left_wrist, left_hip, vertical_point_left]):
                return None
            elbow_flexion_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            torso_lean_angle = calculate_angle(left_hip, left_shoulder, vertical_point_left)
            upper_arm_torso_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
            left_index = [left_wrist[0] + (left_wrist[0] - left_elbow[0]),
                          left_wrist[1] + (left_wrist[1] - left_elbow[1])]
            wrist_angle = calculate_angle(left_elbow, left_wrist, left_index)
            forearm_vertical_angle = calculate_angle(vertical_point_left, left_elbow, left_wrist)
        else:
            if not all([right_shoulder, right_elbow, right_wrist, right_hip, vertical_point_right]):
                return None
            elbow_flexion_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            torso_lean_angle = calculate_angle(right_hip, right_shoulder, vertical_point_right)
            upper_arm_torso_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
            right_index = [right_wrist[0] + (right_wrist[0] - right_elbow[0]),
                           right_wrist[1] + (right_wrist[1] - right_elbow[1])]
            wrist_angle = calculate_angle(right_elbow, right_wrist, right_index)
            forearm_vertical_angle = calculate_angle(vertical_point_right, right_elbow, right_wrist)

        return np.array([
            elbow_flexion_angle,
            torso_lean_angle,
            upper_arm_torso_angle,
            wrist_angle,
            forearm_vertical_angle
        ])
    except Exception as e:
        # print(f"Error calculating angles: {e}")
        return None


# --- 3. MAIN INFERENCE SCRIPT ---


MODEL_PATH = r"C:\Users\LENOVO\Desktop\gym\best_transformer_autoencoder_biceps22.pth"
SCALER_PATH = r"C:\Users\LENOVO\Desktop\gym\pose_scaler_biceps22.pkl"

SEQ_LEN = 30
NUM_FEATURES = 5 # elbow_flexion, torso_lean, upper_arm_torso, wrist, forearm_vertical
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 6


ANOMALY_THRESHOLD = 0.02 

# --- Load Model and Scaler ---
print("Loading model and scaler...")
model = TransformerAutoencoder(NUM_FEATURES, SEQ_LEN, D_MODEL, NHEAD, NUM_LAYERS).to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    print("Please update MODEL_PATH in the script.")
    exit()

model.eval()

try:
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    print(f"ERROR: Scaler file not found at {SCALER_PATH}")
    print("Please update SCALER_PATH in the script.")
    exit()

print("Model and scaler loaded successfully.")

# --- Initialize MediaPipe Pose ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Initialize Webcam ---
# Use 0 for webcam. Or, provide a path to a video file: "path/to/your/test_video.mp4"
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open video source.")
    exit()

# --- Initialize Data Buffer ---
# We need to store the last 30 frames of angle data
data_buffer = deque(maxlen=SEQ_LEN)

# --- Real-time Loop ---
current_status = "Waiting..."
current_error = 0.0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Get frame dimensions
    img_height, img_width, _ = image.shape

    # Process with MediaPipe
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        # --- Our Custom Logic ---
        # 1. Calculate angles
        angles = process_landmarks_to_angles(results.pose_landmarks, img_width, img_height)

        if angles is not None:
            # 2. Scale the angles
            scaled_angles = scaler.transform(angles.reshape(1, -1))

            # 3. Add to buffer
            data_buffer.append(scaled_angles.flatten())

            # 4. Check if buffer is full
            if len(data_buffer) == SEQ_LEN:
                # 5. Prepare window for model
                window = np.array(data_buffer)
                window_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)

                # 6. Get model reconstruction
                with torch.no_grad():
                    recon_out = model(window_t)
                    # Calculate MSE loss (reconstruction error)
                    loss = F.mse_loss(recon_out, window_t)
                    current_error = loss.item()

                # 7. Classify form
                if current_error > ANOMALY_THRESHOLD:
                    current_status = "BAD FORM"
                    color = (0, 0, 255) # Red
                else:
                    current_status = "GOOD FORM"
                    color = (0, 255, 0) # Green

            else:
                current_status = "Initializing..."
                color = (0, 255, 255) # Yellow
        else:
            current_status = "Landmarks not visible"
            color = (0, 0, 255) # Red

    else:
        current_status = "No person detected"
        color = (0, 0, 255) # Red
        data_buffer.clear() # Clear buffer if no person
        current_error = 0.0

    # --- Display Feedback ---
    # Status box
    cv2.rectangle(image, (0, 0), (350, 120), (20, 20, 20), -1)

    # Status Text
    cv2.putText(image, "STATUS:", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, current_status, (100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # Error Text
    cv2.putText(image, "ERROR:", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"{current_error:.6f}", (100, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # Threshold Text
    cv2.putText(image, "THRESH:", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"{ANOMALY_THRESHOLD:.6f}", (100, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


    # Show the image
    cv2.imshow('Biceps Curl AI Coach', image)

    # Press 'q' to quit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Cleanup ---
pose.close()
cap.release()
cv2.destroyAllWindows()
print("Test script finished.")