## pushups onde side only
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

def add_biceps_curl_angles(df):
    df = df.copy()

    elbow_flexion_angle = []
    shoulder_angle = []
    hip_angle =[]
    torso_angle=[]
    wrist_angle = []
    shoulder_elev =[]
    torso_hip_drift =[]




    
    active_arms = []

    for _, row in df.iterrows():

        ## left

        left_shoulder = [row['LEFT_SHOULDER_x'], row['LEFT_SHOULDER_y']]
        left_elbow = [row['LEFT_ELBOW_x'], row['LEFT_ELBOW_y']]
        left_wrist = [row['LEFT_WRIST_x'], row['LEFT_WRIST_y']]
        left_hip = [row.get('LEFT_HIP_x', None), row.get('LEFT_HIP_y', None)]
        left_knee = [row.get('LEFT_KNEE_x', None), row.get('LEFT_KNEE_y', None)]

        ## right

        right_shoulder = [row['RIGHT_SHOULDER_x'], row['RIGHT_SHOULDER_y']]
        right_elbow = [row['RIGHT_ELBOW_x'], row['RIGHT_ELBOW_y']]
        right_wrist = [row['RIGHT_WRIST_x'], row['RIGHT_WRIST_y']]
        right_hip = [row.get('RIGHT_HIP_x', None), row.get('RIGHT_HIP_y', None)]
        right_knee = [row.get('RIGHT_KNEE_x', None), row.get('RIGHT_KNEE_y', None)]

        # Active arm (based on visibility)
        LeftElbow_vis = row.get("LEFT_ELBOW_visibility", 0)
        RightElbow_vis = row.get("RIGHT_ELBOW_visibility", 0)
        active_arm = "left" if LeftElbow_vis > RightElbow_vis else "right"
        active_arms.append(active_arm)

        # Reference vertical
        vertical_point_left = [left_hip[0], left_hip[1] + 1]
        vertical_point_right = [right_hip[0], right_hip[1] + 1]

        # Reference horizontal
        horizontal_point_left = [left_wrist[0]+1, left_wrist[1]]
        horizontal_point_right = [right_wrist[0]+1, right_wrist[1]]

        # =============== LEFT ACTIVE ===============
        if active_arm == "left":

            elbow_flexion_angle.append(calculate_angle(left_shoulder, left_elbow, left_wrist))  # elbow tmm
            shoulder_angle.append(calculate_angle(left_elbow, left_shoulder, left_hip))
            hip_angle.append(calculate_angle(left_shoulder, left_hip, left_knee))
            torso_angle.append(calculate_angle(left_shoulder, left_hip, vertical_point_left))
            wrist_angle.append(calculate_angle(left_elbow, left_wrist, horizontal_point_left))
            shoulder_elev.append(left_shoulder[1]-left_hip[1])
            torso_hip_drift.append(left_hip[0]-left_shoulder[0])


        # =============== RIGHT ACTIVE ===============
        else:

            elbow_flexion_angle.append(calculate_angle(right_shoulder, right_elbow, right_wrist))  # elbow tmm
            shoulder_angle.append(calculate_angle(right_elbow, right_shoulder, right_hip))
            hip_angle.append(calculate_angle(right_shoulder, right_hip, right_knee))
            torso_angle.append(calculate_angle(right_shoulder, right_hip, vertical_point_right))
            wrist_angle.append(calculate_angle(right_elbow, right_wrist, horizontal_point_right))
            shoulder_elev.append(right_shoulder[1]-right_hip[1])
            torso_hip_drift.append(right_hip[0]-right_shoulder[0])

           

    # Add final columns
    df['active_arm'] = active_arms
    df['elbow_flexion_angle'] = elbow_flexion_angle
    df['shoulder_angle'] = shoulder_angle
    df['hip_angle'] = hip_angle
    df['torso_angle'] = torso_angle
    df['wrist_angle'] = wrist_angle
    df['shoulder_elev'] = shoulder_elev
    df['torso_hip_drift'] = torso_hip_drift
   

    return df


# ============================================================
# Keep only active_* landmarks
# ============================================================

def select_active_landmarks(df):
    df = df.copy()

    active_cols = [
        "SHOULDER", "ELBOW", "WRIST",
        "PINKY", "INDEX", "THUMB", "HIP", "KNEE", "ANKLE", "FOOT_INDEX","HEEL"
    ]

    for part in active_cols:
        for suf in ["_x", "_y", "_visibility"]:
            df[f"active_{part.lower()}{suf}"] = np.nan

    # fill
    for i, row in df.iterrows():
        prefix = "LEFT_" if row["active_arm"] == "left" else "RIGHT_"

        for part in active_cols:
            for suf in ["_x", "_y", "_visibility"]:
                old = prefix + part + suf
                new = f"active_{part.lower()}{suf}"
                if old in df.columns:
                    df.at[i, new] = row.get(old, np.nan)

    return df


# ============================================================
# DROP all left/right columns
# ============================================================

def drop_original_landmarks(df):
    return df.drop(columns=[c for c in df.columns if c.startswith("LEFT_") or c.startswith("RIGHT_")],
                   errors="ignore")


# ============================================================
# Load data
# ============================================================

file_path = r"pushups_data_all.csv"

df = pd.read_csv(file_path)

df = add_biceps_curl_angles(df)
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

    #KNEE 
    'LEFT_KNEE_x', 'LEFT_KNEE_y', 'LEFT_KNEE_visibility',
    'RIGHT_KNEE_x', 'RIGHT_KNEE_y', 'RIGHT_KNEE_visibility',

    #ANKLE
    'LEFT_ANKLE_x', 'LEFT_ANKLE_y', 'LEFT_ANKLE_visibility',
    'RIGHT_ANKLE_x', 'RIGHT_ANKLE_y', 'RIGHT_ANKLE_visibility',

    #FOOT
    'LEFT_FOOT_INDEX_x', 'LEFT_FOOT_INDEX_y', 'LEFT_FOOT_INDEX_visibility',
    'RIGHT_FOOT_INDEX_x', 'RIGHT_FOOT_INDEX_y', 'RIGHT_FOOT_INDEX_visibility',
    #HEEL
    'LEFT_HEEL_x', 'LEFT_HEEL_y', 'LEFT_HEEL_visibility',
    'RIGHT_HEEL_x', 'RIGHT_HEEL_y', 'RIGHT_HEEL_visibility',

    # ANGLES
    'elbow_flexion_angle',
    'shoulder_angle',
    'hip_angle',
    'torso_angle',
    'wrist_angle',
    'shoulder_elev',
    'torso_hip_drift',

    # ACTIVE ARM (needed temporarily)
    'active_arm'

]]

df = select_active_landmarks(df)
df = drop_original_landmarks(df)

df = df.sort_values(["video_name", "frame"]).reset_index(drop=True)

df = df.drop(columns=["active_arm"])   
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
print("lllllllllllllllllllllllllllllllllllllllllllllll")
print(num_features)
print(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")

BEST_MODEL_PATH = r"best_transformer_autoencoder_pushups.pth"

SCALER_PATH = r"pose_scaler_pushups.pkl"
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
        print("🔥 Saved BEST model")


# Save final model + scaler
torch.save(model.state_dict(), FINAL_MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print("DONE.")
