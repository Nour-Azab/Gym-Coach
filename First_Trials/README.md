# 🏋️ Week 3 Progress – Pose Landmark Feature Extraction
---
## 🎯 Goal

The goal of Week 3 was to **extract structured features from squat videos** using MediaPipe Pose, in order to prepare training data for a future **sequence model** (LSTM/Transformer). This week is focused on **data preprocessing**, not full classification.

---

## ✅ Achievements This Week

* Implemented **pose detection pipeline** with **MediaPipe + OpenCV**.
* Extracted **33 landmarks per frame**, each containing `(x, y, visibility)`.
* Exported results into a **CSV dataset** with per-frame information.
* Stored processed **frames with skeleton overlay** for debugging/validation.
* Designed a **visibility-based leg selection rule** (choose the clearer leg for analysis).
* Computed the **hip–knee angle relative to vertical**, a key biomechanical feature for squats.
* Added a **simple phase labeler** (`s1`, `s2`, `s3`) using angle thresholds (baseline rule-based labeling).

---

## 📂 Current Output

* **`pose_frames/`** → annotated frames.
* **`pose_landmarks.csv`** → structured dataset.

Example (simplified):

| frame | LEFT\_HIP\_x | LEFT\_HIP\_y | LEFT\_HIP\_visibility | ... | phase |
| ----- | ------------ | ------------ | --------------------- | --- | ----- |
| 15    | 322          | 410          | 0.89                  | ... | s2    |

---
Phase 1
---
![frame_0112](https://github.com/user-attachments/assets/1ae08c69-0ce7-4cc2-9d13-d4c103040d58)

---

Phase 2
---
![frame_0079](https://github.com/user-attachments/assets/ab05c8e2-39e2-427a-8c49-defe7d8b06ba)

---

Phase 3
---

![frame_0246](https://github.com/user-attachments/assets/0f4dd5c5-0063-44a4-a7d9-716f0dd053a0)

---

CSV FILE
---
<img width="2485" height="998" alt="image" src="https://github.com/user-attachments/assets/22d4b2b0-0ad7-4f9f-b982-fcbb3aa240ed" />
---
---

# 🏋️ Week 4–5 Progress – Data Refinement & Baseline Modeling

---

##  **Goal**

The focus of Weeks 4 and 5 was to **improve dataset quality** and begin **testing baseline sequence models** to evaluate the feasibility of squat phase classification from pose data.

---

##  **Achievements**

* Discovered that **around 90% of the Kaggle dataset** contained **poor-quality samples** (incorrect poses, cropped frames, or inconsistent labeling), making it unsuitable for reliable model training.
* **Collected ~50 additional squat videos** from diverse online sources to increase data variability.
* **Recorded ~10 real-life videos** from friends to ensure the dataset reflects **real-world movement distributions** similar to those expected in production.
* Trained a **baseline Autoencoder Transformer model** on a small subset of the data (**15 videos**) to test feasibility.

  * Achieved **promising preliminary results**, demonstrating the model’s ability to capture motion patterns.
* Identified **key issues** affecting performance:

  * Lack of engineered biomechanical features (angles, distances, etc.).
  * Presence of irrelevant or redundant raw pose features.
  * Limited dataset coverage during the pilot test.

---

##  **Next Steps**

* Perform **feature engineering** to include:

  * Joint **angles**, **distances**, and **ratios** between key points.
  * Removal of low-visibility or non-informative landmarks.
* **Expand the dataset** by integrating all cleaned and verified videos into the main CSV file.
* Retrain and evaluate the Transformer model with **engineered features** and the **full dataset**.

---










