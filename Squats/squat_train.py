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

    knee_angles=[]
    hip_angles=[]
    torso_angles=[]
    ankle_angles=[]




    
    active_arms = []

    for _, row in df.iterrows():

        ## left

        left_shoulder = [row['LEFT_SHOULDER_x'], row['LEFT_SHOULDER_y']]
        left_hip = [row['LEFT_HIP_x'], row['LEFT_HIP_y']]
        left_knee = [row['LEFT_KNEE_x'], row['LEFT_KNEE_y']]
        left_ankle = [row['LEFT_ANKLE_x'], row['LEFT_ANKLE_y']]
        left_foot_index = [row['LEFT_FOOT_INDEX_x'], row['LEFT_FOOT_INDEX_y']]

        ## right

        right_shoulder = [row['RIGHT_SHOULDER_x'], row['RIGHT_SHOULDER_y']]
        right_hip = [row['RIGHT_HIP_x'], row['RIGHT_HIP_y']]
        right_knee = [row['RIGHT_KNEE_x'], row['RIGHT_KNEE_y']]
        right_ankle = [row['RIGHT_ANKLE_x'], row['RIGHT_ANKLE_y']]
        right_foot_index = [row['RIGHT_FOOT_INDEX_x'], row['RIGHT_FOOT_INDEX_y']]

        
        # Active arm (based on visibility)
        LeftKnee_vis = row.get("LEFT_KNEE_visibility", 0)
        RightKnee_vis = row.get("RIGHT_KNEE_visibility", 0)
        active_arm = "left" if LeftKnee_vis > RightKnee_vis else "right"
        active_arms.append(active_arm)

        # Reference vertical
        vertical_point_left = [left_hip[0], left_hip[1] - 1]
        vertical_point_right = [right_hip[0], right_hip[1] - 1]

       

        # =============== LEFT ACTIVE ===============
        if active_arm == "left":

            
            knee_angles.append(calculate_angle(left_hip, left_knee, left_ankle))
            hip_angles.append(calculate_angle(left_shoulder, left_hip, left_knee))
            torso_angles.append(calculate_angle(left_shoulder, left_hip, vertical_point_left))
            ankle_angles.append(calculate_angle(left_knee, left_ankle, left_foot_index))


        # =============== RIGHT ACTIVE ===============
        else:

            knee_angles.append(calculate_angle(right_hip, right_knee, right_ankle))
            hip_angles.append(calculate_angle(right_shoulder, right_hip, right_knee))
            torso_angles.append(calculate_angle(right_shoulder, right_hip, vertical_point_right))
            ankle_angles.append(calculate_angle(right_knee, right_ankle, right_foot_index))

            

           

    # Add final columns
    df['active_arm'] = active_arms
    df['knee_angle'] = knee_angles
    df['hip_angle'] = hip_angles
    df['torso_angle'] = torso_angles
    df['ankle_angle'] = ankle_angles
   

    return df


# ============================================================
# Keep only active_* landmarks
# ============================================================

def select_active_landmarks(df):
    df = df.copy()

    active_cols = [
        "SHOULDER", "HIP", "KNEE", "ANKLE", "FOOT_INDEX","HEEL"
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

file_path = r"squats_all.csv"

df = pd.read_csv(file_path)

df = add_biceps_curl_angles(df)
df = df[[
    "video_name", "frame",

    # SHOULDERS
    'LEFT_SHOULDER_x', 'LEFT_SHOULDER_y', 'LEFT_SHOULDER_visibility',
    'RIGHT_SHOULDER_x', 'RIGHT_SHOULDER_y', 'RIGHT_SHOULDER_visibility',


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
    'knee_angle', 'hip_angle', 'torso_angle', 'ankle_angle',

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


BEST_MODEL_PATH = r"best_transformer_autoencoder_squats.pth"
SCALER_PATH = r"pose_scaler_squats.pkl"
FINAL_MODEL_PATH = r"transformer_autoencoder_squats.pth"

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
