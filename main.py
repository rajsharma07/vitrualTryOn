import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

# -------------------------------
# MediaPipe Pose & Segmentation
# -------------------------------
mp_pose = mp.solutions.pose
mp_seg = mp.solutions.selfie_segmentation
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=True)
segmenter = mp_seg.SelfieSegmentation(model_selection=1)





# Helper to display resized images
def show_resized(win_name, img, max_w=900, max_h=700):
    if img is None:
        return
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(win_name, img)
def detect_torso(image, binary_mask, pose_result, margin=0.25, min_vis=0.5):
    h, w = image.shape[:2]
    lm = pose_result.pose_landmarks.landmark

    def lm_safe(lm):
        if lm.visibility < min_vis:
            return None
        return int(lm.x * w), int(lm.y * h)

    ids = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP
    ]

    pts = [lm_safe(lm[i.value]) for i in ids]
    pts = [p for p in pts if p]

    if len(pts) < 3:
        return None, None

    xs, ys = zip(*pts)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    box_w = x_max - x_min
    box_h = y_max - y_min

    x1 = max(int(x_min - box_w * margin), 0)
    x2 = min(int(x_max + box_w * margin), w)
    y1 = max(int(y_min - box_h * 0.35), 0)
    y2 = min(int(y_max + box_h * 0.15), h)

    torso_mask = binary_mask[y1:y2, x1:x2]
    torso = image[y1:y2, x1:x2] * torso_mask[:, :, None]

    return (x1, y1, x2, y2), torso

#---------------------------------
def recommend_shirt_size(pose_result, img_w, img_h):
    lm = pose_result.pose_landmarks.landmark

    def pt(idx):
        return np.array([
            lm[idx].x * img_w,
            lm[idx].y * img_h
        ])

    # Key landmarks
    LS = pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
    RS = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    LH = pt(mp_pose.PoseLandmark.LEFT_HIP)
    RH = pt(mp_pose.PoseLandmark.RIGHT_HIP)

    # Measurements (pixels)
    shoulder_width = np.linalg.norm(LS - RS)
    shoulder_center = (LS + RS) / 2
    hip_center = (LH + RH) / 2
    torso_length = np.linalg.norm(shoulder_center - hip_center)

    hip_width = np.linalg.norm(LH - RH)

    # ----------------------------
    # Normalized ratios
    # ----------------------------
    torso_ratio = torso_length / shoulder_width
    hip_ratio = hip_width / shoulder_width

    # ----------------------------
    # Size classification (MEN – TSHIRT)
    # ----------------------------
    if shoulder_width < 140:
        size = "S"
    elif shoulder_width < 170:
        size = "M"
    elif shoulder_width < 200:
        size = "L"
    else:
        size = "XL"

    # Fit type (bonus)
    if torso_ratio < 1.25:
        fit = "Short Torso"
    elif torso_ratio > 1.5:
        fit = "Long Torso"
    else:
        fit = "Regular Fit"

    return {
        "size": size,
        "shoulder_px": int(shoulder_width),
        "torso_px": int(torso_length),
        "torso_ratio": round(torso_ratio, 2),
        "fit": fit
    }


# Load shirt image
cloth = None
base = Path(__file__).resolve().parent
candidates = [base / "tshirt.png", base / "pictures" / "tshirt.png", Path("tshirt.png"), Path("pictures") / "tshirt.png"]
found_path = None
for p in candidates:
    if p.exists():
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            cloth = img
            found_path = p
            break

if cloth is None:
    raise FileNotFoundError(f"tshirt.png not found. Tried: {[str(p) for p in candidates]}")

if cloth.ndim == 2:
    cloth = cv2.cvtColor(cloth, cv2.COLOR_GRAY2BGRA)
elif cloth.shape[2] == 3:
    b, g, r = cv2.split(cloth)
    alpha = np.full(b.shape, 255, dtype=b.dtype)
    cloth = cv2.merge([b, g, r, alpha])
elif cloth.shape[2] != 4:
    raise ValueError("tshirt.png must have 1, 3 or 4 channels")


# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    img_h, img_w = frame.shape[:2]
    
    pose_result = pose.process(image_rgb)
    seg_result = segmenter.process(image_rgb)

    if not pose_result.pose_landmarks:
        cv2.putText(frame, "No pose detected", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        show_resized("Virtual Try-On", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    binary_mask = (seg_result.segmentation_mask > 0.5).astype(np.uint8)

    annotated = frame.copy()
    mp_draw.draw_landmarks(
        annotated,
        pose_result.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

    bbox, torso = detect_torso(frame, binary_mask, pose_result)

    if bbox is not None:
        cloth_rgb = cloth[:, :, :3]
        cloth_alpha = cloth[:, :, 3] / 255.0

        lm = pose_result.pose_landmarks.landmark

        l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        if l_sh.visibility >= 0.6 and r_sh.visibility >= 0.6:
            lx, ly = int(l_sh.x * img_w), int(l_sh.y * img_h)
            rx, ry = int(r_sh.x * img_w), int(r_sh.y * img_h)

            shoulder_width = abs(rx - lx)

            scale_factor = 1.75
            target_w = int(shoulder_width * scale_factor)
            aspect = cloth_rgb.shape[0] / cloth_rgb.shape[1]
            target_h = int(target_w * aspect)

            cloth_rgb_resized = cv2.resize(cloth_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
            cloth_alpha_resized = cv2.resize(cloth_alpha, (target_w, target_h), interpolation=cv2.INTER_AREA)

            x_center = (lx + rx) // 2
            y_top = min(ly, ry) - int(0.17 * target_h)

            x1 = x_center - target_w // 2
            y1 = y_top
            x2 = x1 + target_w
            y2 = y1 + target_h

            if x1 < 0:
                cloth_rgb_resized = cloth_rgb_resized[:, -x1:]
                cloth_alpha_resized = cloth_alpha_resized[:, -x1:]
                x1 = 0

            if y1 < 0:
                cloth_rgb_resized = cloth_rgb_resized[-y1:, :]
                cloth_alpha_resized = cloth_alpha_resized[-y1:, :]
                y1 = 0

            x2 = min(x1 + cloth_rgb_resized.shape[1], img_w)
            y2 = min(y1 + cloth_rgb_resized.shape[0], img_h)

            cloth_rgb_resized = cloth_rgb_resized[:y2 - y1, :x2 - x1]
            cloth_alpha_resized = cloth_alpha_resized[:y2 - y1, :x2 - x1]

            torso_mask_full = np.zeros((img_h, img_w), dtype=np.uint8)
            tx1, ty1, tx2, ty2 = bbox
            torso_mask_full[ty1:ty2, tx1:tx2] = binary_mask[ty1:ty2, tx1:tx2]

            roi = annotated[y1:y2, x1:x2]
            roi_mask = torso_mask_full[y1:y2, x1:x2]

            for c in range(3):
                roi[:, :, c] = (
                    roi_mask * (
                        cloth_alpha_resized * cloth_rgb_resized[:, :, c] +
                        (1 - cloth_alpha_resized) * roi[:, :, c]
                    ) +
                    (1 - roi_mask) * roi[:, :, c]
                )

            annotated[y1:y2, x1:x2] = roi

    # Get body info for display
    body_info = recommend_shirt_size(pose_result, img_w, img_h)

    # Display size and fit information on frame
    label = f"Size: {body_info['size']} | Fit: {body_info['fit']}"
    info_text = f"Shoulders: {body_info['shoulder_px']}px | Torso: {body_info['torso_px']}px"
    ratio_text = f"Torso Ratio: {body_info['torso_ratio']}"

    # Background box for text
    cv2.rectangle(annotated, (10, 10), (600, 110), (0, 0, 0), -1)
    
    # Display text
    cv2.putText(annotated, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated, info_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(annotated, ratio_text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)

    # Display frame
    show_resized("Virtual Try-On", annotated)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
segmenter.close()
