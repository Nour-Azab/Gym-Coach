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




