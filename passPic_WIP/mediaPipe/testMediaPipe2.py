
import cv2
import numpy as np
import mediapipe as mp

mp_selfie = mp.solutions.selfie_segmentation

img = cv2.imread(r"C:\Users\S10DIGITAL\python\py_GIMP\openCV\Pic14.jpeg")
orig_h, orig_w = img.shape[:2]
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Run MediaPipe
with mp_selfie.SelfieSegmentation(model_selection=1) as segment:
    res = segment.process(rgb)

mask_mp = res.segmentation_mask  # shape e.g. 354x272

# --- FIX: resize MP mask to original image dimensions ---
mask_mp_resized = cv2.resize(mask_mp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

# Visualize
overlay = img.copy()
contour = (mask_mp_resized > 0.5).astype(np.uint8) * 255
contour_edges = cv2.Canny(contour, 50, 150)
overlay[contour_edges > 0] = (0, 0, 255)  # red outline

# Save result
cv2.imwrite(r"C:\Users\S10DIGITAL\python\py_GIMP\openCV\mp_overlay_check.png", overlay)

print("Saved overlay to mp_overlay_check.png")
