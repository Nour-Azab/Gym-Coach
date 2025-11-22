import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.model_selection import train_test_split

import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")




def calculate_angle(a, b, c):
  
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    dot_product = np.dot(ba, bc)
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine_angle = np.clip(dot_product / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))


def add_biceps_curl_angles(df):

    df = df.copy()

    # Angle lists
    elbow_flexion_angle = []
    torso_lean_angle = []
    upper_arm_torso_angle = []
    wrist_angle = []
    forearm_vertical_angle = []
    active_arms = []

    for _, row in df.iterrows():
        # Landmarks
        left_shoulder = [row['LEFT_SHOULDER_x'], row['LEFT_SHOULDER_y']]
        left_elbow = [row['LEFT_ELBOW_x'], row['LEFT_ELBOW_y']]
        left_wrist = [row['LEFT_WRIST_x'], row['LEFT_WRIST_y']]
        left_hip = [row.get('LEFT_HIP_x', None), row.get('LEFT_HIP_y', None)]

        right_shoulder = [row['RIGHT_SHOULDER_x'], row['RIGHT_SHOULDER_y']]
        right_elbow = [row['RIGHT_ELBOW_x'], row['RIGHT_ELBOW_y']]
        right_wrist = [row['RIGHT_WRIST_x'], row['RIGHT_WRIST_y']]
        right_hip = [row.get('RIGHT_HIP_x', None), row.get('RIGHT_HIP_y', None)]

        # Determine active arm
        LeftElbow_vis = row.get("LEFT_ELBOW_visibility", 0)
        RightElbow_vis = row.get("RIGHT_ELBOW_visibility", 0)
        active_arm = "left" if LeftElbow_vis > RightElbow_vis else "right"
        active_arms.append(active_arm)

        # Vertical reference points
        vertical_point_left  = [left_shoulder[0],  left_shoulder[1]  - 100]
        vertical_point_right = [right_shoulder[0], right_shoulder[1] - 100]

        # Compute angles for active arm only
        if active_arm == "left":
            elbow_flexion_angle.append(calculate_angle(left_shoulder, left_elbow, left_wrist))
            torso_lean_angle.append(calculate_angle(left_hip, left_shoulder, vertical_point_left) if left_hip[0] else 0.0)
            upper_arm_torso_angle.append(calculate_angle(left_hip, left_shoulder, left_elbow) if left_hip[0] else 0.0)
            left_index = [left_wrist[0] + (left_wrist[0] - left_elbow[0]),
                          left_wrist[1] + (left_wrist[1] - left_elbow[1])]
            wrist_angle.append(calculate_angle(left_elbow, left_wrist, left_index))
            forearm_vertical_angle.append(calculate_angle(vertical_point_left, left_elbow, left_wrist))
        else:
            elbow_flexion_angle.append(calculate_angle(right_shoulder, right_elbow, right_wrist))
            torso_lean_angle.append(calculate_angle(right_hip, right_shoulder, vertical_point_right) if right_hip[0] else 0.0)
            upper_arm_torso_angle.append(calculate_angle(right_hip, right_shoulder, right_elbow) if right_hip[0] else 0.0)
            right_index = [right_wrist[0] + (right_wrist[0] - right_elbow[0]),
                           right_wrist[1] + (right_wrist[1] - right_elbow[1])]
            wrist_angle.append(calculate_angle(right_elbow, right_wrist, right_index))
            forearm_vertical_angle.append(calculate_angle(vertical_point_right, right_elbow, right_wrist))

    # Add only active arm columns
    df_active = df[['frame', 'video_name']].copy()
    df_active['active_arm'] = active_arms
    df_active['elbow_flexion_angle'] = elbow_flexion_angle
    df_active['torso_lean_angle'] = torso_lean_angle
    df_active['upper_arm_torso_angle'] = upper_arm_torso_angle
    df_active['wrist_angle'] = wrist_angle
    df_active['forearm_vertical_angle'] = forearm_vertical_angle

    return df_active


    # Load and preprocess data
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

file_path = "/kaggle/input/ggggggg/lasttttttttttttttttttt_biceps_lastttttttttttttttttttttttttttttt.xlsx"
sheet_name = "rwaddd2"

df = pd.read_excel(file_path, sheet_name=sheet_name)
print(f"Initial shape: {df.shape}")



df = add_biceps_curl_angles(df)
df = df.sort_values(by=["video_name", "frame"]).reset_index(drop=True)

# Drop face landmarks AND lower body landmarks (keep upper body only)
columns_to_drop = [
    # Face landmarks
    'LEFT_EYE_INNER_x', 'LEFT_EYE_INNER_y', 'LEFT_EYE_INNER_visibility',
    'LEFT_EYE_x', 'LEFT_EYE_y', 'LEFT_EYE_visibility',
    'LEFT_EYE_OUTER_x', 'LEFT_EYE_OUTER_y', 'LEFT_EYE_OUTER_visibility',
    'RIGHT_EYE_INNER_x', 'RIGHT_EYE_INNER_y', 'RIGHT_EYE_INNER_visibility',
    'RIGHT_EYE_x', 'RIGHT_EYE_y', 'RIGHT_EYE_visibility',
    'RIGHT_EYE_OUTER_x', 'RIGHT_EYE_OUTER_y', 'RIGHT_EYE_OUTER_visibility',
    'MOUTH_LEFT_x', 'MOUTH_LEFT_y', 'MOUTH_LEFT_visibility',
    'MOUTH_RIGHT_x', 'MOUTH_RIGHT_y', 'MOUTH_RIGHT_visibility',

    'LEFT_KNEE_x', 'LEFT_KNEE_y', 'LEFT_KNEE_visibility',
    'RIGHT_KNEE_x', 'RIGHT_KNEE_y', 'RIGHT_KNEE_visibility',
    'LEFT_ANKLE_x', 'LEFT_ANKLE_y', 'LEFT_ANKLE_visibility',
    'RIGHT_ANKLE_x', 'RIGHT_ANKLE_y', 'RIGHT_ANKLE_visibility',
    'LEFT_HEEL_x', 'LEFT_HEEL_y', 'LEFT_HEEL_visibility',
    'RIGHT_HEEL_x', 'RIGHT_HEEL_y', 'RIGHT_HEEL_visibility',
    'LEFT_FOOT_INDEX_x', 'LEFT_FOOT_INDEX_y', 'LEFT_FOOT_INDEX_visibility',
    'RIGHT_FOOT_INDEX_x', 'RIGHT_FOOT_INDEX_y', 'RIGHT_FOOT_INDEX_visibility'
]

# Only drop columns that exist in the dataframe
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df.drop(columns=existing_columns_to_drop, inplace=True)

print(f"Dropped {len(existing_columns_to_drop)} columns (face + lower body)")
print(f"Shape after dropping: {df.shape}")

# Define metadata columns (no phase or rep_counter)
meta_cols = ["frame", "video_name", "active_arm"]
pose_cols = [c for c in df.columns if c not in meta_cols]

# Scale numeric features
scaler = StandardScaler()
df[pose_cols] = scaler.fit_transform(df[pose_cols])

def create_windows_for_video(data, window_size=30, stride=5):
    X = []
    for i in range(0, len(data) - window_size, stride):
        X.append(data[i:i+window_size])
    return np.array(X)

X_all = []
for _, group in df.groupby("video_name"):
    X_tmp = create_windows_for_video(group[pose_cols].values, window_size=30, stride=5)
    if len(X_tmp) > 0:  # skip empty videos
        X_all.append(X_tmp)

# Only concatenate non-empty arrays
X_all = np.concatenate(X_all, axis=0)
print(f"Size of windowed data: X={X_all.shape}")

X_train, X_temp = train_test_split(X_all, test_size=0.2, random_state=42)
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, X_train_t), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, X_val_t), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, X_test_t), batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


seq_len = X_train.shape[1]
num_features = X_train.shape[2]

print(f"Sequence Length: {seq_len}")
print(f"Number of Features: {num_features}")

model = TransformerAutoencoder(num_features, seq_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
mse_loss = nn.MSELoss()
epochs = 30

train_losses = []
val_losses = []
best_val_loss = float('inf')
# ===============================
# Save path in /kaggle/working
# ===============================
BEST_MODEL_PATH = "/kaggle/working/best_transformer_autoencoder_biceps22.pth"
FINAL_MODEL_PATH = "/kaggle/working/transformer_autoencoder_biceps22.pth"
SCALER_PATH = "/kaggle/working/pose_scaler_biceps22.pkl"

best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for x_batch, _ in train_loader:
        x_batch = x_batch.to(device)

        optimizer.zero_grad()
        recon_out = model(x_batch)
        loss = mse_loss(recon_out, x_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_val, _ in val_loader:
            x_val = x_val.to(device)
            recon_out = model(x_val)
            loss = mse_loss(recon_out, x_val)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"✅ New best model saved at /kaggle/working! Val Loss: {best_val_loss:.4f}")

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# Save final model and scaler
torch.save(model.state_dict(), FINAL_MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"\n✅ Final model saved at: {FINAL_MODEL_PATH}")
print(f" Scaler saved at: {SCALER_PATH}")
print(f"Model expects {num_features} features per frame")