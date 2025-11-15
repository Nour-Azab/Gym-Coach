import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import joblib
import sys

# ---------------- Configuration ----------------
VIDEO_PATH = 0  
WINDOW_SIZE = 30
NUM_LANDMARKS = 17 
NUM_ANGLES = 5    
NUM_FEATURES = NUM_LANDMARKS * 3 + NUM_ANGLES 
ANOMALY_THRESHOLD = 0.03 


SCALER_PATH = r"C:\Users\Abdallah\Desktop\pose_scaler_biceps.pkl"
MODEL_PATH  = r"C:\Users\Abdallah\Desktop\transformer_autoencoder_biceps.pth"
# --------------------------


LANDMARK_MAP = {
    'NOSE': 0,
    'LEFT_EAR': 7,
    'RIGHT_EAR': 8,
    'LEFT_SHOULDER': 11,
    'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13,
    'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15,
    'RIGHT_WRIST': 16,
    'LEFT_PINKY': 17,
    'RIGHT_PINKY': 18,
    'LEFT_INDEX': 19,
    'RIGHT_INDEX': 20,
    'LEFT_THUMB': 21,
    'RIGHT_THUMB': 22,
    'LEFT_HIP': 23,
    'RIGHT_HIP': 24
}
# ---------------- Utility Functions ----------------

def calculate_angle(a, b, c):
    """Calculates the angle in degrees between three points (a, b, c), where 'b' is the vertex."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def extract_biceps_angles(landmarks):
    """
    Extracts the 5 key angles for biceps curls.
    Uses .get() for safety, returning 0 if a landmark is missing.
    """
    left_shoulder = [landmarks.get('LEFT_SHOULDER_x', 0), landmarks.get('LEFT_SHOULDER_y', 0)]
    left_elbow = [landmarks.get('LEFT_ELBOW_x', 0), landmarks.get('LEFT_ELBOW_y', 0)]
    left_wrist = [landmarks.get('LEFT_WRIST_x', 0), landmarks.get('LEFT_WRIST_y', 0)]

    right_shoulder = [landmarks.get('RIGHT_SHOULDER_x', 0), landmarks.get('RIGHT_SHOULDER_y', 0)]
    right_elbow = [landmarks.get('RIGHT_ELBOW_x', 0), landmarks.get('RIGHT_ELBOW_y', 0)]
    right_wrist = [landmarks.get('RIGHT_WRIST_x', 0), landmarks.get('RIGHT_WRIST_y', 0)]

    nose = [landmarks.get('NOSE_x', 0), landmarks.get('NOSE_y', 0)]

    # Calculate angles
    elbow_flexion_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
    elbow_flexion_right = calculate_angle(right_shoulder, right_elbow, right_wrist)

    shoulder_stability_left = calculate_angle(left_elbow, left_shoulder, nose)
    shoulder_stability_right = calculate_angle(right_elbow, right_shoulder, nose)

    shoulder_horizontal = calculate_angle(left_shoulder, nose, right_shoulder)

    return [elbow_flexion_left, elbow_flexion_right,
            shoulder_stability_left, shoulder_stability_right,
            shoulder_horizontal]

def build_feature_vector(landmarks):
    """Builds the 56-feature vector in the correct order for the scaler."""
    features = []
    
    lm_names = [
        'NOSE', 'LEFT_EAR', 'RIGHT_EAR',
        'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
        'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
        'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
        'LEFT_HIP', 'RIGHT_HIP'
    ]
    
 
    for lm in lm_names:
      
        features.extend([
            landmarks.get(f'{lm}_x', 0),
            landmarks.get(f'{lm}_y', 0),
            landmarks.get(f'{lm}_visibility', 0)
        ])
        

    angles = extract_biceps_angles(landmarks)
    features.extend(angles)
    
  
    return np.array(features)

# ---------------- Model  ----------------

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
        out = self.decoder(z, memory)
        return self.output_proj(out)

# ---------------- Initialization ----------------
print("Loading models and scaler...")
try:
    scaler = joblib.load(SCALER_PATH)
    print(f"Scaler loaded from {SCALER_PATH}")
except FileNotFoundError:
    print(f"Error: Scaler file not found at {SCALER_PATH}")
    sys.exit()
except Exception as e:
    print(f"Error loading scaler: {e}")
    sys.exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model = TransformerAutoencoder(num_features=NUM_FEATURES, seq_len=WINDOW_SIZE).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    sys.exit()
except RuntimeError as e:
    print(f"RuntimeError loading model. This usually means a feature mismatch.")
    print(f"Current model expects {NUM_FEATURES} features.")
    print(f"Error details: {e}")
    sys.exit()
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()
    
model.eval()
print("Model and scaler loaded successfully.")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video source {VIDEO_PATH}")
    pose.close()
    sys.exit()

pose_buffer = []
frame_count = 0
form_status = "Analyzing..."
status_color = (255, 255, 255) # White
current_error = 0.0

# ---------------- Main Loop ----------------
print("Starting video capture. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or video file.")
        break
        
    frame_count += 1
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )


    if results.pose_landmarks:
        landmarks = {}
        for lm_name, lm_index in LANDMARK_MAP.items():
            if len(results.pose_landmarks.landmark) > lm_index:
                lm = results.pose_landmarks.landmark[lm_index]
                landmarks[f'{lm_name}_x'] = lm.x
                landmarks[f'{lm_name}_y'] = lm.y
                landmarks[f'{lm_name}_visibility'] = lm.visibility
            else:
                landmarks[f'{lm_name}_x'] = 0
                landmarks[f'{lm_name}_y'] = 0
                landmarks[f'{lm_name}_visibility'] = 0

        fv = build_feature_vector(landmarks)
        pose_buffer.append(fv)
    else:
        pose_buffer.append(np.zeros(NUM_FEATURES))

    if len(pose_buffer) > WINDOW_SIZE:
        pose_buffer = pose_buffer[-WINDOW_SIZE:]

    # --- Inference ---
    if len(pose_buffer) == WINDOW_SIZE:
        window = np.array(pose_buffer)
        
        if np.isnan(window).any() or np.isinf(window).any():
            print("Warning: NaN or Inf detected in window, skipping frame.")
            form_status = "Error: Invalid data"
            status_color = (0, 0, 255)
            pose_buffer.pop(0)
            continue

        try:
            window_scaled = scaler.transform(window)
        except ValueError as e:
            print(f"Error during scaling: {e}. Check feature consistency.")
       
            form_status = "Error: Scaling"
            status_color = (0, 0, 255)
            pose_buffer.pop(0)
            continue
            
        window_tensor = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(device)

      
        with torch.no_grad():
            recon = model(window_tensor)
            error = torch.mean((window_tensor - recon)**2).item()
            current_error = error # Store for display

    
        if error > ANOMALY_THRESHOLD:
            form_status = "❌ Poor Form"
            status_color = (0, 0, 255) 
        else:
            form_status = "✅ Good Form"
            status_color = (0, 255, 0)

    # --- Draw Info Panel ---

    cv2.rectangle(frame, (0, 0), (300, 110), (0, 0, 0), -1)
    
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Error: {current_error:.4f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Form: {form_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)

    cv2.imshow("Biceps Curl Analyzer", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("'q' pressed. Exiting...")
        break

# ---------------- Cleanup ----------------
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
pose.close()
print("Done.")