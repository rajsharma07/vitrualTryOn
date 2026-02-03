import cv2
import numpy as np

img = cv2.imread("../DataSet/processed_clothes/shirt_01.png", cv2.IMREAD_UNCHANGED)  # RGBA
bgr = img[:, :, :3].copy()
alpha = img[:, :, 3]

mask = (alpha > 0).astype(np.uint8) * 255

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

shirt_contour = max(contours, key=cv2.contourArea)

pts = shirt_contour.reshape(-1, 2)

top    = pts[np.argmin(pts[:,1])]
bottom = pts[np.argmax(pts[:,1])]
left   = pts[np.argmin(pts[:,0])]
right  = pts[np.argmax(pts[:,0])]

mid_y = int((top[1] + bottom[1]) / 2)

sleeve_pts = pts[np.abs(pts[:,1] - mid_y) < 5]

left_sleeve  = sleeve_pts[np.argmin(sleeve_pts[:,0])]
right_sleeve = sleeve_pts[np.argmax(sleeve_pts[:,0])]

top_band = pts[pts[:,1] < top[1] + 40]

left_collar  = top_band[np.argmin(top_band[:,0])]
right_collar = top_band[np.argmax(top_band[:,0])]

for p in [top, bottom, left, right]:
    cv2.circle(bgr, tuple(p), 6, (0, 0, 255), -1)

cv2.imshow("Keypoints", bgr)
cv2.waitKey(0)
