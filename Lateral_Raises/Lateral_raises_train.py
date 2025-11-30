import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


# ============================================================
# Angle calculation
# ============================================================

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    dot_product = np.dot(ba, bc)
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine_angle = np.clip(dot_product / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))


# ================================================================
# Add angles + detect active arm
# =================================================================

def add_lateral_raises_angles(df):
    df = df.copy()

    left_shoulder_angles, right_shoulder_angles = [], []
    left_elbow_angles, right_elbow_angles = [], []
    left_torso_angles, right_torso_angles = [], []
    left_shoulder_elevs, right_shoulder_elevs = [], []
    left_wrist_angles, right_wrist_angles = [], []
    left_arm_drifts, right_arm_drifts = [], []


    for _, row in df.iterrows():
        # --- Left landmarks ---
        left_shoulder = [row['LEFT_SHOULDER_x'], row['LEFT_SHOULDER_y']]
        left_elbow = [row['LEFT_ELBOW_x'], row['LEFT_ELBOW_y']]
        left_wrist = [row['LEFT_WRIST_x'], row['LEFT_WRIST_y']]
        left_hip = [row['LEFT_HIP_x'], row['LEFT_HIP_y']]

        # --- Right landmarks ---
        right_shoulder = [row['RIGHT_SHOULDER_x'], row['RIGHT_SHOULDER_y']]
        right_elbow = [row['RIGHT_ELBOW_x'], row['RIGHT_ELBOW_y']]
        right_wrist = [row['RIGHT_WRIST_x'], row['RIGHT_WRIST_y']]
        right_hip = [row['RIGHT_HIP_x'], row['RIGHT_HIP_y']]

        # --- Calculate angles ---
        # Shoulder Angle (Arm Abduction: Elbow, Shoulder, Hip)
        left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
        right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)

        # Elbow Angle (Shoulder, Elbow, Wrist)
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Torso Angle (Shoulder, Hip, Vertical) - Measures torso lean
        left_hip_vertical = [left_hip[0], left_hip[1] + 1]
        right_hip_vertical = [right_hip[0], right_hip[1] + 1]
        left_torso_angle = calculate_angle(left_shoulder, left_hip, left_hip_vertical)
        right_torso_angle = calculate_angle(right_shoulder, right_hip, right_hip_vertical)

         # --- Avoid Shoulder Shrugging ---
        left_shoulder_elevation = left_shoulder[1] - left_hip[1]
        right_shoulder_elevation = right_shoulder[1] - right_hip[1]

        # --- Wrist Alignment Angle (Elbow → Wrist → Horizontal) ---
        left_wrist_horizontal = [left_wrist[0] + 1, left_wrist[1]]
        right_wrist_horizontal = [right_wrist[0] + 1, right_wrist[1]]

        left_wrist_angle = calculate_angle(left_elbow, left_wrist, left_wrist_horizontal)
        right_wrist_angle = calculate_angle(right_elbow, right_wrist, right_wrist_horizontal)

        # --- Arm Forward/Backward Drift ---
        left_arm_drift = left_elbow[0] - left_shoulder[0]
        right_arm_drift = right_elbow[0] - right_shoulder[0]


        # --- Append to lists ---
        left_shoulder_angles.append(left_shoulder_angle)
        right_shoulder_angles.append(right_shoulder_angle)
        left_elbow_angles.append(left_elbow_angle)
        right_elbow_angles.append(right_elbow_angle)
        left_torso_angles.append(left_torso_angle)
        right_torso_angles.append(right_torso_angle)
        left_shoulder_elevs.append(left_shoulder_elevation)
        right_shoulder_elevs.append(right_shoulder_elevation)
        left_wrist_angles.append(left_wrist_angle)
        right_wrist_angles.append(right_wrist_angle)
        left_arm_drifts.append(left_arm_drift)
        right_arm_drifts.append(right_arm_drift)



    # --- Add new columns ---
    df['left_shoulder_angle'] = left_shoulder_angles
    df['right_shoulder_angle'] = right_shoulder_angles
    df['left_elbow_angle'] = left_elbow_angles
    df['right_elbow_angle'] = right_elbow_angles
    df['left_torso_angle'] = left_torso_angles
    df['right_torso_angle'] = right_torso_angles
    df['left_shoulder_elevation'] = left_shoulder_elevs
    df['right_shoulder_elevation'] = right_shoulder_elevs
    df['left_wrist_angle'] = left_wrist_angles
    df['right_wrist_angle'] = right_wrist_angles
    df['left_arm_drift'] = left_arm_drifts
    df['right_arm_drift'] = right_arm_drifts


    return df

# ============================================================
# Load data
# ============================================================

file_path = r"pose_landmarks_all_lateral_raises.csv"

df = pd.read_csv(file_path)

df = add_lateral_raises_angles(df)
df = df[[
    "video_name", "frame",

    # SHOULDERS
    'LEFT_SHOULDER_x', 'LEFT_SHOULDER_y', 'LEFT_SHOULDER_visibility',
    'RIGHT_SHOULDER_x', 'RIGHT_SHOULDER_y', 'RIGHT_SHOULDER_visibility',

    # ELBOWS
    'LEFT_ELBOW_x', 'LEFT_ELBOW_y', 'LEFT_ELBOW_visibility',
    'RIGHT_ELBOW_x', 'RIGHT_ELBOW_y', 'RIGHT_ELBOW_visibility',

    # WRISTS
    'LEFT_WRIST_x', 'LEFT_WRIST_y', 'LEFT_WRIST_visibility',
    'RIGHT_WRIST_x', 'RIGHT_WRIST_y', 'RIGHT_WRIST_visibility',

    # PINKY
    'LEFT_PINKY_x', 'LEFT_PINKY_y', 'LEFT_PINKY_visibility',
    'RIGHT_PINKY_x', 'RIGHT_PINKY_y', 'RIGHT_PINKY_visibility',

    # INDEX
    'LEFT_INDEX_x', 'LEFT_INDEX_y', 'LEFT_INDEX_visibility',
    'RIGHT_INDEX_x', 'RIGHT_INDEX_y', 'RIGHT_INDEX_visibility',

    # THUMB
    'LEFT_THUMB_x', 'LEFT_THUMB_y', 'LEFT_THUMB_visibility',
    'RIGHT_THUMB_x', 'RIGHT_THUMB_y', 'RIGHT_THUMB_visibility',

    # HIPS
    'LEFT_HIP_x', 'LEFT_HIP_y', 'LEFT_HIP_visibility',
    'RIGHT_HIP_x', 'RIGHT_HIP_y', 'RIGHT_HIP_visibility',


    # ANGLES
    'left_shoulder_angle',
    'right_shoulder_angle',
    'left_elbow_angle',
    'right_elbow_angle',
    'left_torso_angle',
    'right_torso_angle',
    'left_shoulder_elevation',
    'right_shoulder_elevation',
    'left_wrist_angle',
    'right_wrist_angle',
    'left_arm_drift',
    'right_arm_drift'

    
]]


df = df.sort_values(["video_name", "frame"]).reset_index(drop=True)  
df=df.drop(columns=["frame"])


# ============================================================
# Scaling
# ============================================================
pose_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Number of features:", len(pose_cols))
print("Feature names in order:")
for i, c in enumerate(pose_cols):
    print(f"{i}: {c}")

scaler = StandardScaler()
df[pose_cols] = scaler.fit_transform(df[pose_cols])


# ============================================================
# Create windows
# ============================================================

def create_windows_for_video(data, window_size=30, stride=5):
    X = []
    for i in range(0, len(data) - window_size + 1, stride):  # FIXED
        X.append(data[i:i+window_size])
    return np.array(X)


X_all = []
for _, g in df.groupby("video_name"):
    Xg = create_windows_for_video(g[pose_cols].values)
    if len(Xg) > 0:
        X_all.append(Xg)

X_all = np.concatenate(X_all)


# ============================================================
# Split
# ============================================================

X_train, X_temp = train_test_split(X_all, test_size=0.2, random_state=42)
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, X_val), batch_size=32)
test_loader  = DataLoader(TensorDataset(X_test, X_test), batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Model
# ============================================================

class TransformerAutoencoder(nn.Module):
    def __init__(self, num_features, seq_len):
        super().__init__()
        d_model = 128
        nhead = 8
        num_layers = 6

        self.input_proj = nn.Linear(num_features, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.output_proj = nn.Linear(d_model, num_features)

    def forward(self, x):
        z = self.input_proj(x)
        mem = self.encoder(z)
        out = self.decoder(z, mem)
        return self.output_proj(out)


seq_len = X_train.shape[1]
num_features = X_train.shape[2]

model = TransformerAutoencoder(num_features, seq_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

BEST_MODEL_PATH = r"best_lateral_raises_transformer_autoencoder.pth"

SCALER_PATH = r"pose_scaler_lateral_raises.pkl"
FINAL_MODEL_PATH = r"transformer_autoencoder_pushups.pth"

# ============================================================
# Train loop
# ============================================================

best_val = 999999

for epoch in range(30):
    model.train()
    train_loss = 0

    for xb, _ in train_loader:
        xb = xb.to(device)
        optimizer.zero_grad()
        rec = model(xb)
        loss = criterion(rec, xb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            rec = model(xb)
            val_loss += criterion(rec, xb).item()

    t = train_loss / len(train_loader)
    v = val_loss / len(val_loader)

    print(f"Epoch {epoch+1} | Train {t:.4f} | Val {v:.4f}")

    if v < best_val:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_val = v
        print("Saved BEST model")


# Save final model + scaler
torch.save(model.state_dict(), FINAL_MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print("DONE.")