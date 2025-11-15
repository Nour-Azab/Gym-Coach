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


def add_biceps_curl_angles(df):
    """
    Adds the essential angles for biceps curl analysis with full descriptive names:
    - elbow_flexion_angle_left
    - elbow_flexion_angle_right
    - shoulder_stability_angle_left
    - shoulder_stability_angle_right
    - shoulder_horizontal_angle
    """
    df = df.copy()

    # Lists to store angles
    elbow_flexion_angle_left = []
    elbow_flexion_angle_right = []
    shoulder_stability_angle_left = []
    shoulder_stability_angle_right = []
    shoulder_horizontal_angle = []

    for _, row in df.iterrows():
        # --- Landmarks ---
        left_shoulder = [row['LEFT_SHOULDER_x'], row['LEFT_SHOULDER_y']]
        left_elbow = [row['LEFT_ELBOW_x'], row['LEFT_ELBOW_y']]
        left_wrist = [row['LEFT_WRIST_x'], row['LEFT_WRIST_y']]
        left_ear = [row.get('LEFT_EAR_x', None), row.get('LEFT_EAR_y', None)]
       # nose = [row['NOSE_x'], row['NOSE_y']]

        right_shoulder = [row['RIGHT_SHOULDER_x'], row['RIGHT_SHOULDER_y']]
        right_elbow = [row['RIGHT_ELBOW_x'], row['RIGHT_ELBOW_y']]
        right_wrist = [row['RIGHT_WRIST_x'], row['RIGHT_WRIST_y']]
        right_ear = [row.get('RIGHT_EAR_x', None), row.get('RIGHT_EAR_y', None)]

        # --- Elbow Flexion Angle (Shoulder - Elbow - Wrist) ---
        elbow_flexion_angle_left_val = calculate_angle(left_shoulder, left_elbow, left_wrist)
        elbow_flexion_angle_right_val = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # --- Shoulder Stability Angle (Elbow - Shoulder - Head/Ear) ---
      # left_head = left_ear if left_ear[0] is not None else nose
       # right_head = right_ear if right_ear[0] is not None else nose

       # shoulder_stability_angle_left_val = calculate_angle(left_elbow, left_shoulder, left_head)
      #  shoulder_stability_angle_right_val = calculate_angle(right_elbow, right_shoulder, right_head)

        # --- Shoulder Horizontal Angle (Left Shoulder - Nose - Right Shoulder) ---
      #  shoulder_horizontal_angle_val = calculate_angle(left_shoulder, nose, right_shoulder)

        # --- Append ---
        elbow_flexion_angle_left.append(elbow_flexion_angle_left_val)
        elbow_flexion_angle_right.append(elbow_flexion_angle_right_val)
      #  shoulder_stability_angle_left.append(shoulder_stability_angle_left_val)
        #shoulder_stability_angle_right.append(shoulder_stability_angle_right_val)
     #   shoulder_horizontal_angle.append(shoulder_horizontal_angle_val)

    # --- Add to DataFrame ---
    df['elbow_flexion_angle_left'] = elbow_flexion_angle_left
    df['elbow_flexion_angle_right'] = elbow_flexion_angle_right
    df['shoulder_stability_angle_left'] = shoulder_stability_angle_left
    df['shoulder_stability_angle_right'] = shoulder_stability_angle_right
  #  df['shoulder_horizontal_angle'] = shoulder_horizontal_angle

    return df

# Load and preprocess data
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

file_path = r"C:\Users\Abdallah\Desktop\lasttttttttttttttttttt_biceps.xlsx"
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
    
    # Lower body landmarks (not needed for lateral raises)
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
meta_cols = ["frame", "video_name"]
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
BEST_MODEL_PATH = r"D:\llllllll\best_transformer_autoencoder_biceps.pth"
torch.save(model.state_dict(), BEST_MODEL_PATH)

for epoch in range(epochs):
    model.train()
    train_loss, train_correct, total = 0, 0, 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        class_out = model(x_batch)

        recon_out = model(x_batch)
        loss = mse_loss(recon_out, x_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for x_val, _ in val_loader:
            x_val = x_val.to(device)
            recon_out = model(x_val)
            loss = mse_loss(recon_out, x_val)
            val_loss += loss.item()

    avg_train_loss = train_loss/len(train_loader)
    avg_val_loss = val_loss/len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    
    # --- The Logic to Save the Best Model ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"✅ New best model saved! Val Loss: {best_val_loss:.4f}")

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f}")
    
    # Save model and scaler
torch.save(model.state_dict(), "transformer_autoencoder_biceps.pth")
joblib.dump(scaler, "pose_scaler_biceps_raises.pkl")
print("\nAutoencoder model saved as: transformer_autoencoder_biceps.pth")
print("Scaler saved as: pose_scaler_biceps.pkl")
print(f"Model expects {num_features} features per frame")

torch.save(model.state_dict(), "transformer_autoencoder_biceps.pth")
joblib.dump(scaler, "pose_scaler_biceps.pkl")
print("Autoencoder model and scaler saved.")