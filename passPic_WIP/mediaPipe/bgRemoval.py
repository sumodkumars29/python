import cv2
import mediapipe as mp
import numpy as np
import sys
import os

# CONFIGURE INPUT/OUTPUT PATH
if len(sys.argv) <2:
    print("Usage: 'python bgRemoval.py <image_path>'")
    sys.exit(1)

input_path = sys.argv[1]

if not os.path.exists(input_path):
    print(f"File does not exist: {input_path}")
    sys.exit(1)
 
# Create output filename
base_name = os.path.basename(input_path)
name, ext = os.path.splitext(base_name)
output_path = os.path.join(os.path.dirname(input_path), f"{name}_bgEdited.jpg")

# Load an image
img = cv2.imread(input_path)
if img is None:
    print("Failed to load image.")
    sys.exit(1)

# ----------------------------
# MEIDAPIPE SEGMENTATION
# ----------------------------

mp_selfie_segmentation = mp.solutions.selfie_segmentation


with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie:
    result = selfie.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    mask = result.segmentation_mask

# Convert mask to visible form
# mask_vis = (mask * 255).astype("uint8")


# POST PROCESS MASK
# Threshold to binary (confident foreground/background)
_, mask_bin = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
# mask_bin = mask_bin.astype(np.uint8)
mask_bin = 1 - mask_bin

# SMOOTH EDGES
mask_smooth = cv2.GaussianBlur(mask.astype(np.float32), (5,5), 0)
mask_smooth[mask_smooth < 0.1] = 0  # remove tiny background noise
mask_smooth[mask_smooth > 0.9] = 1  # fully keep confident subject pixels
# MAKE 3 CHANNEL MASK FOR BLENDING
mask_3c = cv2.merge([mask_smooth]*3) # still 0-1 float

# ----------------------------
# APPLY MASK TO IMAGE with WHITE BACKGROUND
# ----------------------------
white_bg = np.ones_like(img, dtype=np.float32) * 255    # white background
foreground = img.astype(np.float32) * mask_3c + white_bg * (1 - mask_3c)
foreground = foreground.astype(np.uint8)

# Save result
cv2.imwrite(output_path, foreground)
print(f"Saved image with white background: {output_path}")


