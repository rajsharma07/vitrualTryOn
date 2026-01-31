import cv2
import mediapipe as mp
import numpy as np
import json
import math

# -------------------------------
# Load shirt and anchors
# -------------------------------
shirt = cv2.imread(
    "DataSet/processed_clothes/shirt_01.png",
    cv2.IMREAD_UNCHANGED
)

if shirt is None:
    raise RuntimeError("Shirt image not found")

if shirt.shape[2] != 4:
    raise RuntimeError("Shirt must be a PNG with alpha channel")

with open("DataSet/anchors/shirt_01.json") as f:
    anchors = json.load(f)

cloth_ls = np.array(anchors["left_shoulder"], dtype=np.float32)
cloth_rs = np.array(anchors["right_shoulder"], dtype=np.float32)
cloth_bottom = np.array(anchors["bottom"], dtype=np.float32)

cloth_width = np.linalg.norm(cloth_rs - cloth_ls)
cloth_mid = (cloth_ls + cloth_rs) / 2

# -------------------------------
# MediaPipe Pose
# -------------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# -------------------------------
# Overlay helper
# -------------------------------
def overlay_transparent(bg, overlay, x, y):
    bg_h, bg_w = bg.shape[:2]
    h, w = overlay.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, bg_w), min(y + h, bg_h)

    if x1 >= x2 or y1 >= y2:
        return bg

    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    crop = overlay[oy1:oy2, ox1:ox2]
    alpha = crop[:, :, 3] / 255.0
    alpha = alpha[:, :, None]

    bg[y1:y2, x1:x2] = (
        alpha * crop[:, :, :3] +
        (1 - alpha) * bg[y1:y2, x1:x2]
    ).astype(np.uint8)

    return bg

# -------------------------------
# Geometry helpers
# -------------------------------
def rotate_point(pt, center, angle_deg):
    angle = math.radians(angle_deg)
    ox, oy = center
    px, py = pt

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return np.array([qx, qy])

# -------------------------------
# Smoothing
# -------------------------------
prev_x = prev_y = prev_angle = prev_scale = None
smooth_alpha = 0.8

# -------------------------------
# Main loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        l_px = np.array([
            lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
            lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
        ])
        r_px = np.array([
            lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w,
            lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h
        ])

        user_mid = (l_px + r_px) / 2

        # -------------------------------
        # Scale (smoothed)
        # -------------------------------
        scale = np.linalg.norm(r_px - l_px) / cloth_width
        scale = np.clip(scale, 0.6, 1.8)

        if prev_scale is None:
            scale_s = scale
        else:
            scale_s = smooth_alpha * prev_scale + (1 - smooth_alpha) * scale
        prev_scale = scale_s
        scale = scale_s

        new_w = int(shirt.shape[1] * scale)
        new_h = int(shirt.shape[0] * scale)

        shirt_resized = cv2.resize(shirt, (new_w, new_h))

        # -------------------------------
        # Rotation (clamped + smoothed)
        # -------------------------------
        dx = r_px[0] - l_px[0]
        dy = l_px[1] - r_px[1]
        angle = math.degrees(math.atan2(dy, dx))
        angle = np.clip(angle, -25, 25)

        if prev_angle is None:
            angle_s = angle
        else:
            angle_s = smooth_alpha * prev_angle + (1 - smooth_alpha) * angle
        prev_angle = angle_s

        img_center = np.array([new_w / 2, new_h / 2])

        M = cv2.getRotationMatrix2D(tuple(img_center), angle_s, 1.0)
        shirt_rotated = cv2.warpAffine(
            shirt_resized,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # -------------------------------
        # Correct anchor positioning
        # -------------------------------
        cloth_mid_scaled = cloth_mid * scale
        cloth_bottom_scaled = cloth_bottom * scale

        rotated_mid = rotate_point(
            cloth_mid_scaled,
            img_center,
            angle_s
        )

        shirt_drop = cloth_bottom_scaled[1] - cloth_mid_scaled[1]

        raw_x = user_mid[0] - rotated_mid[0]
        raw_y = user_mid[1] - rotated_mid[1] + 0.08 * shirt_drop

        if prev_x is None:
            x, y = raw_x, raw_y
        else:
            x = smooth_alpha * prev_x + (1 - smooth_alpha) * raw_x
            y = smooth_alpha * prev_y + (1 - smooth_alpha) * raw_y

        prev_x, prev_y = x, y
        x, y = int(x), int(y)

        frame = overlay_transparent(frame, shirt_rotated, x, y)

    cv2.imshow("Virtual Try-On", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
