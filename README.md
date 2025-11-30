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

To train Coachy’s models, we needed high-quality videos of people performing each exercise **correctly**.

We collected our dataset through **manual web scraping**:

* We visited many training and workout websites
* Downloaded videos where the exercise form was **accurate and professional**
* Manually filtered every video to remove incorrect, low-quality, or noisy samples

After final filtering, each exercise had its own clean set of videos ready for pose extraction.

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


