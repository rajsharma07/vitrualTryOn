import cv2
import mediapipe as mp
import numpy as np

# ============================
# Utility
# ============================
def show_resized(win_name, img, max_w=900, max_h=700):
    if img is None:
        return
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(win_name, img)


# ============================
# Load Image
# ============================
image = cv2.imread("person4.jpeg")
if image is None:
    raise FileNotFoundError("person.jpg not found")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_h, img_w = image.shape[:2]


# ============================
# MediaPipe Init
# ============================
mp_pose = mp.solutions.pose
mp_seg = mp.solutions.selfie_segmentation
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=True)
segmenter = mp_seg.SelfieSegmentation(model_selection=1)


# ============================
# Pose + Segmentation
# ============================
pose_result = pose.process(image_rgb)
seg_result = segmenter.process(image_rgb)

if not pose_result.pose_landmarks:
    raise RuntimeError("Pose not detected")

binary_mask = (seg_result.segmentation_mask > 0.5).astype(np.uint8)


# ============================
# Draw Landmarks
# ============================
annotated = image.copy()
mp_draw.draw_landmarks(
    annotated,
    pose_result.pose_landmarks,
    mp_pose.POSE_CONNECTIONS
)

# show_resized("Pose Landmarks", annotated)
# cv2.waitKey(0)


# ============================
# Torso Detection
# ============================
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


bbox, torso = detect_torso(image, binary_mask, pose_result)

if bbox is None:
    raise RuntimeError("Torso detection failed")

# show_resized("Torso Masked", torso)
# cv2.waitKey(0)


# ============================
# Load Cloth (PNG)
# ============================
cloth = cv2.imread("tshirt.png", cv2.IMREAD_UNCHANGED)
if cloth is None or cloth.shape[2] != 4:
    raise ValueError("tshirt.png must be RGBA")

cloth_rgb = cloth[:, :, :3]
cloth_alpha = cloth[:, :, 3] / 255.0


# ============================
# Shoulder-Based Placement
# ============================
lm = pose_result.pose_landmarks.landmark

l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

if l_sh.visibility < 0.6 or r_sh.visibility < 0.6:
    raise RuntimeError("Shoulders not clearly visible")

lx, ly = int(l_sh.x * img_w), int(l_sh.y * img_h)
rx, ry = int(r_sh.x * img_w), int(r_sh.y * img_h)

shoulder_width = abs(rx - lx)

# ✔ Correct scale
scale_factor = 1.75
target_w = int(shoulder_width * scale_factor)
aspect = cloth_rgb.shape[0] / cloth_rgb.shape[1]
target_h = int(target_w * aspect)

cloth_rgb = cv2.resize(cloth_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
cloth_alpha = cv2.resize(cloth_alpha, (target_w, target_h), interpolation=cv2.INTER_AREA)

x_center = (lx + rx) // 2
y_top = min(ly, ry) - int(0.17 * target_h)

x1 = x_center - target_w // 2
y1 = y_top
x2 = x1 + target_w
y2 = y1 + target_h


# ============================
# Boundary Fix
# ============================
if x1 < 0:
    cloth_rgb = cloth_rgb[:, -x1:]
    cloth_alpha = cloth_alpha[:, -x1:]
    x1 = 0

if y1 < 0:
    cloth_rgb = cloth_rgb[-y1:, :]
    cloth_alpha = cloth_alpha[-y1:, :]
    y1 = 0

x2 = min(x1 + cloth_rgb.shape[1], img_w)
y2 = min(y1 + cloth_rgb.shape[0], img_h)

cloth_rgb = cloth_rgb[:y2 - y1, :x2 - x1]
cloth_alpha = cloth_alpha[:y2 - y1, :x2 - x1]


# ============================
# Torso Mask Constraint
# ============================
torso_mask_full = np.zeros((img_h, img_w), dtype=np.uint8)
tx1, ty1, tx2, ty2 = bbox
torso_mask_full[ty1:ty2, tx1:tx2] = binary_mask[ty1:ty2, tx1:tx2]

roi = image[y1:y2, x1:x2]
roi_mask = torso_mask_full[y1:y2, x1:x2]

for c in range(3):
    roi[:, :, c] = (
        roi_mask * (
            cloth_alpha * cloth_rgb[:, :, c] +
            (1 - cloth_alpha) * roi[:, :, c]
        ) +
        (1 - roi_mask) * roi[:, :, c]
    )

image[y1:y2, x1:x2] = roi




# ============================
# Body Measurement & Size Estimation
# ============================
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
    
    
body_info = recommend_shirt_size(pose_result, img_w, img_h)

print("Recommended Size:", body_info["size"])
print("Torso Ratio:", body_info["torso_ratio"])

# ============================
# Final Output
# ============================
show_resized("Virtual Try-On Result", image)
cv2.imwrite("tryon_result.jpg", image)
cv2.waitKey(0)

pose.close()
cv2.destroyAllWindows()


# ============================
# Display Size on Image
# ============================
label = f"Recommended Size: {body_info['size']} ({body_info['fit']})"

cv2.rectangle(image, (20, 20), (480, 90), (0, 0, 0), -1)
cv2.putText(
    image,
    label,
    (30, 65),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 255, 0),
    2,
    cv2.LINE_AA
)
