# 🏋️‍♂️ Coachy – AI Gym Coach

## 1️⃣ Introduction

**Coachy** is an AI-powered virtual gym coach designed to put a **complete gym experience inside every home**.

The concept is simple:
You open the camera — from your laptop, phone, or any device — perform your exercise, and **Coachy gives you instant live feedback** about your form:

* *“Lower your hips.”*
* *“Extend your arms fully.”*
* *“Keep your elbows tucked.”*

Coachy currently supports **four core exercises**:
**Squats, Push-ups, Biceps Curls, and Lateral Raises.**

Our goal is to give everyone a **smart, accessible, and real-time** coaching tool without needing any equipment or special setup.

---

## 2️⃣ Data Collection

To train Coachy’s models, we needed high-quality videos of people performing each exercise **correctly and consistently**.

Our data collection process had two main parts:

### **1. Manual Web Scraping**

* We visited multiple training and workout websites
* Downloaded videos where the exercise form was **clean and correct**
* Manually filtered all samples to keep only high-quality movements
* Removed bad form, occlusions, incorrect angles, and noisy videos

### **2. Recording Our Own Videos (Very Important Step)**

To make sure the model sees the **same distribution of camera angles, lighting, distances, and real-world noise** that it will face on the website, we also:

* Recorded videos using **our own mobile phones**
* Asked our friends to perform the exercises and contributed many samples
* Ensured that the videos matched the typical conditions users will have (home, normal lighting, normal camera distance)

This step was **crucial**, because it helped the model generalize to real users — not just professional studio footage.
Special thanks to all our friends who helped us build a more robust and realistic dataset ❤️.Sure — here are **clean, professional dataset summary tables for *each model*** (each exercise).
You can paste this directly into your README under the “Data Collection” or “Feature Engineering” section.

---

---


## 3️⃣ Feature Engineering

### 3.1 Extracting Pose Landmarks

Each video was fed into **MediaPipe Pose**, which outputs the full set of **33 landmarks** (x, y, visibility) per frame.

For each exercise:

1. We extracted all frames
2. Stored the 33 landmarks in a CSV per video
3. Merged all CSVs into one dataset per exercise
4. Selected only the landmarks relevant to the movement (not all 33 are needed)

---

## 3.2 Landmark Selection per Exercise

Different exercises activate different joints.
Below is the exact table of landmarks used for each exercise.
(All include: **x, y, visibility**)

---

### 🟦 **Squats**

| Side      | Landmarks Used                                                                   |
| --------- | -------------------------------------------------------------------------------- |
| **LEFT**  | LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX, LEFT_HEEL       |
| **RIGHT** | RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX, RIGHT_HEEL |

---

### 🟥 **Biceps Curls**

| Side      | Landmarks Used                                                                             |
| --------- | ------------------------------------------------------------------------------------------ |
| **LEFT**  | LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_PINKY, LEFT_INDEX, LEFT_THUMB, LEFT_HIP        |
| **RIGHT** | RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_PINKY, RIGHT_INDEX, RIGHT_THUMB, RIGHT_HIP |

---

### 🟩 **Push-ups**

| Side      | Landmarks Used                                                                                                                              |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **LEFT**  | LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_PINKY, LEFT_INDEX, LEFT_THUMB, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT, LEFT_HEEL            |
| **RIGHT** | RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_PINKY, RIGHT_INDEX, RIGHT_THUMB, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT, RIGHT_HEEL |

---

### 🟨 **Lateral Raises**

| Both Sides                                                                                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST, LEFT_PINKY, RIGHT_PINKY, LEFT_INDEX, RIGHT_INDEX, LEFT_THUMB, RIGHT_THUMB, LEFT_HIP, RIGHT_HIP |

---

## 3.3 Angle & Distance Features (Major Improvement)

After the first attempt using raw landmarks only, we added **biomechanical angle features** (and some distances).
This dramatically improved all models.

### ⭐ Lateral Raises – Angle Features

```
left_shoulder_angle  
right_shoulder_angle  
left_elbow_angle  
right_elbow_angle  
left_torso_angle  
right_torso_angle  
left_shoulder_elevation  
right_shoulder_elevation  
left_wrist_angle  
right_wrist_angle  
left_arm_drift  
right_arm_drift  
```

---

### ⭐ Push-ups – Angle Features

```
active_arm  
elbow_flexion_angle  
shoulder_angle  
hip_angle  
torso_angle  
wrist_angle  
shoulder_elev  
torso_hip_drift  
```

---

### ⭐ Biceps Curls – Angle Features

```
active_arm  
elbow_flexion_angle  
forearm_vertical_angle  
wrist_angle  
upper_arm_torso_angle  
torso_lean_angle  
```

---

### ⭐ Squats – Angle Features

```
active_arm  
knee_angles  
hip_angles  
torso_angles  
ankle_angles  
```

---

## 3.4 One-Side Active Arm Selection

Finally, we improved performance by using **only one side of the body** instead of both.

* If the **left** side has higher visibility → use **all left landmarks**
* If the **right** side is clearer → use **all right landmarks**

This removes the noise of the side that is partially hidden from the camera.

This upgrade made the model more stable, more accurate, and reduced unnecessary variance.

---

# 📊 **Dataset Summary per Exercise**

Below are the datasets used to train each exercise-specific model in Coachy.

---

## 🟦 **Push-Ups Dataset**

| Item                 | Description                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------- |
| **Source**           | Web-scraped videos + mobile-recorded videos from friends                                                |
| **Total Videos**     | *56*                                                                                                    |
| **Frames Extracted** | *21800*                                                                                                 |
| **Landmarks Used**   | Shoulder, Elbow, Wrist, Pinky, Index, Thumb, Hip, Knee, Ankle, Foot, Heel (Left or Right only)          |
| **Angle Features**   | Elbow flexion, shoulder angle, hip angle, torso angle, wrist angle, shoulder elevation, torso-hip drift |
| **Window Size**      | 30 frames                                                                                               |
| **Stride**           | 5                                                                                                       |
| **Labels**           | Unsupervised (only good form)                                                                           |
| **Purpose**          | Anomaly detection for bad push-up form                                                                  |

---

## 🟥 **Squats Dataset**

| Item                 | Description                                                       |
| -------------------- | ----------------------------------------------------------------- |
| **Source**           | Web-scraped videos + mobile-recorded videos from friends          |
| **Total Videos**     | *94*                                                              |
| **Frames Extracted** | *42400*                                                           |
| **Landmarks Used**   | Shoulder, Hip, Knee, Ankle, Foot Index, Heel (Left or Right only) |
| **Angle Features**   | Knee angles, hip angles, torso angles, ankle angles               |
| **Window Size**      | 30 frames                                                         |
| **Stride**           | 5                                                                 |
| **Labels**           | Unsupervised (only good form)                                     |
| **Purpose**          | Detect shallow squats, poor depth, back arching                   |

---

## 🟩 **Biceps Curls Dataset**

| Item                 | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| **Source**           | Web-scraped videos + mobile-recorded videos from friends                                    |
| **Total Videos**     | *65*                                                                                        |
| **Frames Extracted** | *38400*                                                                                     |
| **Landmarks Used**   | Shoulder, Elbow, Wrist, Pinky, Index, Thumb, Hip (Left or Right only)                       |
| **Angle Features**   | Elbow flexion, forearm vertical angle, wrist angle, upper-arm torso angle, torso lean angle |
| **Window Size**      | 30 frames                                                                                   |
| **Stride**           | 5                                                                                           |
| **Labels**           | Unsupervised                                                                                |
| **Purpose**          | Detect partial ROM, swinging, weak contraction                                              |

---

## 🟨 **Lateral Raises Dataset**

| Item                 | Description                                                           |
| -------------------- | --------------------------------------------------------------------- |
| **Source**           | Web-scraped videos + mobile-recorded videos from friends              |
| **Total Videos**     | *67*                                                                  |
| **Frames Extracted** | *28200*                                                               |
| **Landmarks Used**   | Shoulder, Elbow, Wrist, Pinky, Index, Thumb, Hip (both sides needed)  |
| **Angle Features**   | Shoulder elevation, elbow angle, wrist angle, torso angles, arm drift |
| **Window Size**      | 30 frames                                                             |
| **Stride**           | 5                                                                     |
| **Labels**           | Unsupervised                                                          |
| **Purpose**          | Detect wrist mistakes, ROM issues, elbow drop                         |

---
---

## 4️⃣ Model Architecture

To evaluate exercise form, Coachy uses a **Transformer Autoencoder**, trained **only on correct-form videos**.
This allows the system to learn the *normal pattern* of each exercise and detect deviations as **anomalies**, which correspond to bad form.

### ⭐ Why an Autoencoder?

All videos we collected (from websites + our mobile phones) contain **good form only**.
So instead of training a classifier (which needs good vs bad), we train an **autoencoder** that learns:

* how correct movements look
* how correct joints move together
* how correct angles change over time

When the user performs the exercise:

* If the movement is correct → the model reconstructs it well → **low error**
* If the movement is wrong → reconstruction fails → **high error = bad form alert**

This approach is perfect for **anomaly detection on motion sequences**.

---

## 4.1 Input Representation

### ✔️ Features

Each frame contains:

* Selected pose landmarks (x, y, visibility)
* Exercise-specific angle features (like shoulder angle, elbow flexion, torso lean, etc.)
* One-side “active arm” selection to reduce noise

### ✔️ Scaling

All numerical features are standardized using:

```python
StandardScaler()
```

This stabilizes training and prevents landmarks with large value ranges from dominating the loss.

### ✔️ Temporal Windows

Instead of feeding individual frames, we feed **windows of 30 frames** with a **stride of 5**, giving the model a full short motion sequence.

This allows the Transformer to learn the **motion pattern**, not just static posture.

---

## 4.2 Transformer Autoencoder Architecture

Your model architecture:

```
Input (num_features)
      ↓  Linear (input projection)
Encoder → TransformerEncoder (6 layers, 8 heads)
      ↓  
Decoder → TransformerDecoder (6 layers, 8 heads)
      ↓
Output (reconstructed features)
```

### 🔧 Key Components

| Component             | Description                                                           |
| --------------------- | --------------------------------------------------------------------- |
| **Input Projection**  | Maps raw features → 128-dimensional embedding                         |
| **Encoder**           | Learns the correct motion pattern of the exercise                     |
| **Decoder**           | Attempts to reconstruct the original sequence from the encoded memory |
| **Output Projection** | Converts model output back to original feature size                   |

### 📏 Model Hyperparameters

* `d_model = 128`
* `nhead = 8`
* `num_layers = 6` for encoder + decoder
* `optimizer = Adam(lr=1e-4)`
* `loss = MSELoss()`
* `batch_size = 32`
* `epochs = 30`

---

## 4.3 Loss Function

We use **Mean Squared Error (MSE)** between the input sequence and the reconstructed sequence:

```python
criterion = nn.MSELoss()
```

**Reason:**
If the user performs the exercise correctly, the model easily reconstructs the movement → low MSE.
If the user performs it incorrectly (bad angles, wrong depth, leaning, etc.), the reconstruction is poor → high MSE.

This is the core idea behind **anomaly detection**.

---

## 4.4 Why Transformers? (Technical Reasons)

Transformers outperform RNN/LSTM models for motion and time-series for several reasons:

### ✔️ 1. They capture long-range dependencies

Movement in an exercise is not only about the current frame — it's about **how joints move across time**.
Self-attention lets the model compare any frame to any other frame directly.

### ✔️ 2. They handle variable speed

Different people move faster or slower.
Transformers are robust to timing variation.

### ✔️ 3. They are great for multi-joint coordination

Exercise form depends on the relationship between:

* hips
* shoulders
* elbows
* wrists
* knees

Self-attention naturally models these multi-joint interactions.

### ✔️ 4. They are powerful for reconstruction tasks

The encoder learns the "ideal" movement pattern.
The decoder replicates it.
Any deviation → immediate anomaly.

---

## 4.5 Summary

The Transformer Autoencoder architecture allows Coachy to:

* Learn the **correct** motion for each exercise
* Detect **incorrect** or dangerous form
* Provide **instant feedback** during the live session
* Generalize to different users, speeds, and camera setups
* Stay lightweight enough for real-time inference

---


# 5️⃣ Real-Time Feedback Logic

Coachy provides live feedback by combining **two complementary systems**:

### **1. Rule-Based Biomechanics Feedback (Per-Frame)**

We analyzed the most common form mistakes in biomechanics (ROM errors, elbow position, depth, wrist alignment… etc.) and converted them into **real-time rules**.

These rules run **every frame**, so feedback like:

* “Go down more!”
* “Extend your arms!”
* “Keep your back straight!”

…is returned **instantly** the moment the mistake happens.

### **2. Transformer Autoencoder (Anomaly Detection)**

If none of the explicit rules fire, we let the Transformer check the frame window.
If reconstruction error is high → the model detects **bad form** automatically.

This gives us a hybrid system:

| Component       | Detects                                   |
| --------------- | ----------------------------------------- |
| **Rule-based**  | Specific biomechanical mistakes           |
| **Autoencoder** | Any unfamiliar / unusual movement pattern |

Together, this guarantees we catch **all bad forms**, even the ones we didn’t write explicit rules for.

---

# 5.1 Push-Ups – Real-Time Feedback Rules

For push-ups, the key feature is:

```
elbow_flexion_angle = angles[1]
```

The movement is divided into 4 phases based on elbow angle:

| Phase  | Condition                 | Meaning              |
| ------ | ------------------------- | -------------------- |
| **P1** | angle ≥ 150°              | Top / full extension |
| **P2** | angle decreasing (150→65) | Going down           |
| **P3** | angle ≤ 65°               | Bottom position      |
| **P4** | angle increasing (65→150) | Going up             |

### **ROM Rules (Range of Motion)**

Coachy checks:

#### **1. Incomplete Bottom Range**

If user starts going down but doesn’t reach the bottom →
→ **“Go Down More!”**

#### **2. Incomplete Top Range**

If user comes up but doesn’t reach full extension →
→ **“Go Up More!”**

### **3. Anomaly Detection**

If reconstruction error > threshold →
→ **“POOR FORM!”**

### **4. Rep Counting**

A valid rep is:

```
P4 → P1    (Going up → Fully extended)
```

Invalid reps do not increase the counter.

---

# 5.2 Squats – Real-Time Feedback Rules

Key feature:

```
knee_angle = angles[1]
```

Phases:

| Phase  | Condition        |            |
| ------ | ---------------- | ---------- |
| **S1** | angle > 160°     | Standing   |
| **S2** | angle decreasing | Going down |
| **S3** | angle ≤ 90°      | Bottom     |
| **S4** | angle increasing | Going up   |

### **ROM Rules**

#### **1. Not Going Deep Enough**

If user starts descending but never reaches proper bottom →
→ **“Not Going Low Enough!”**

#### **2. Back Arching**

If torso angle exceeds threshold during descent →
→ **“Don’t Arch Your Back!”**

### **3. Anomaly Detection**

If reconstruction error > threshold:
→ **“Bad Form!”**

### **4. Rep Counting**

Valid rep happens on:

```
S4 → S1    (Going up → Standing fully)
```

---

# 5.3 Biceps Curls – Real-Time Feedback Rules

Key angle:

```
elbow_flexion_angle = angles[1]
```

Phases:

| Phase  | Condition        |                       |
| ------ | ---------------- | --------------------- |
| **B1** | angle ≥ 160°     | Rest / Full extension |
| **B2** | angle decreasing | Going up              |
| **B3** | angle ≤ 60°      | Top contraction       |
| **B4** | angle increasing | Going down            |

### **ROM Rules**

#### **1. Not Fully Extending Arms (bottom ROM)**

→ **“Extend your arms more!”**

#### **2. Weak contraction at top**

→ **“Contract your arms more!”**

### **3. Autoencoder Detection**

If reconstruction error high:
→ **“POOR FORM!”**

### **4. Rep Counting**

Valid rep:

```
B4 → B1
```

---

# 5.4 Lateral Raises – Real-Time Feedback Rules

Key angle:

```
shoulder_angle = (angles[0] + angles[1]) / 2
```

Phases:

| Phase   | Condition        |            |
| ------- | ---------------- | ---------- |
| **LR1** | angle ≤ 30°      | Rest       |
| **LR2** | angle increasing | Going up   |
| **LR3** | angle ≥ 75°      | Top        |
| **LR4** | angle decreasing | Going down |

### **ROM Rules**

#### **1. Not Lowering Enough**

→ **“Relax arms at the end!”**

#### **2. Not Raising Enough**

→ **“Raise Elbow!”**

#### **3. Wrist Higher than Elbow**

Biomechanically dangerous for shoulder →
→ **“Wrist higher than elbow!”**

### **4. Autoencoder Detection**

→ **“POOR FORM!”**

### **5. Rep Counting**

Valid rep:

```
LR4 → LR1
```

---

# 5.5 Summary of Feedback System

| Component                 | Role                                  |
| ------------------------- | ------------------------------------- |
| **Angle-based phases**    | Identify motion stage (up/down/rest)  |
| **ROM rules**             | Ensure correct depth and extension    |
| **Joint alignment rules** | Wrist–elbow, back arching, etc.       |
| **Autoencoder error**     | Detects any unusual / unseen movement |
| **Rep counter**           | Tracks valid reps only                |

The result is a **fast, precise, hybrid feedback system** that works frame-by-frame while also analyzing short movement sequences.

---







