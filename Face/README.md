I will make code for Face Detection and specification!

Without any deep learning model, "face_analysis.py" performs real-time face analysis geometrically, using only MediaPipe FaceMesh to extract landmarks for the entire face, not just the eyes. 

## Key Features
+ Real-time head pose estimation: yaw, pitch, roll
+ Eye closure and blink detection using EAR
+ Lightweight and fast, deployable on Jetson Nano
+ No external model or inference engine required 
+ Video Frame -> face recognition -> extract pixel coordinates based on landmark index -> calculation (EAR, yaw/pitch/roll)

## Landmark System
Each index corresponds to specific facial region: eyes, nose, and ears. According to the index map provided by MediaPipe FaceMesh, we assign several index out of 468 values (0~467) to each facial region.

<img src="https://github.com/user-attachments/assets/04d0ad19-b978-4a59-9161-0f1396754886" width="300"/>
(출처: Google AI for Developers)

| Region         | Sample Indexes               |
| -------------- | ---------------------------- |
| Left Eye       | 33, 160, 158, 133, 153, 144  |
| Right Eye      | 362, 385, 387, 263, 373, 380 |
| Nose   | 1                            |
| Left Ear area  | 234                          |
| Right Ear area | 454                          |

## Eye Closure and Blink Detection 
(EAR 이론 간략 설명)
(EAR 이론용 이미지 추가, 출처)
EAR drops below a threshold value (0.2) when eyes are closed. If closed for a sustained time(2 sec.), it's considered eye closure; Short-term drops count as a blink.
```text
EAR = (‖p2 − p6‖ + ‖p3 − p5‖) / (2 × ‖p1 − p4‖) 
(p1~p6 are predefined eye landmarks forming vertical and horizontal distances)
```

## Head Pose Estimation
Instead of deep learning model, head pose estimation is implemented based on simple geometric logic using facial landmark positions. It's an approximate logic-based estimation for lightweight use.

<img src="https://github.com/user-attachments/assets/360a5d8d-d42e-4c4b-bf13-f8c9db18c35b" width="300"/>

+ Yaw (left-right):
Estimate left-right rotation by comparing which side of the ear is closer to the nose
```text
yaw = (‖nose − left_ear‖ − ‖nose − right_ear‖) / width × 100
```
+ Pitch (up-down):
Estimate up-down rotation by comparing eye height and nose height
```text
pitch = (eye_mid_y − nose_y) / height × 100
```
+Roll (tilt):
Estimate head tilt by analyzing the slope of the horizontal line connecting the two eyes
```text
roll = degrees(arctangent(Δy / Δx))
```
