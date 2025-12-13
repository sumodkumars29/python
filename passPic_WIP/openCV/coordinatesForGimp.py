import sys
import cv2
import numpy as np


def fail(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


# Detection parameters
s_Factor = 1.2  # smaller -> more sensitive
m_Neighbours = 6  # larger -> stricter

# Aspect ratio constants (815 x 1063)
targeted_Height_AR = 1.303  # height to width (1063 / 815)
targeted_Width_AR = 0.767  # widht to height (815 / 1063)

# Haar-cascade for face detection
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Verify that the cascade file exists
if face_cascade.empty():
    fail(f"Cannot load cascade classifier from: {cascade_path}")

# Ensure image argument is provided
if len(sys.argv) < 2:
    fail("No image path provided")

image_path = sys.argv[1]

# Ensure argument provided is an image file
if not image_path.lower().endswith((".jpg", ".png", ".jpeg")):
    fail("Not an image")

# Read the image file
img = cv2.imread(image_path)

# If image is not readable
if img is None:
    fail("Invalid image path or unreadable file")

# Capture the image width and height
ImageH, ImageW, _ = img.shape

# Convert to LAB colour space for better colour difference handling
image_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Set sample background (rows 5-80, first 150 columns)
sample_region = image_lab[5:80, 0:150]
mean_bg_color = sample_region.mean(axis=(0, 1))

# Scan from top downward
threshold = 15  # LAB is more sensitive, hence lower threshold
top_of_head = 0

# Scan only the central 80% columns to avoid edge noise
x_start_scan = int(ImageW * 0.1)
x_end_scan = int(ImageW * 0.9)

for y in range(ImageH):
    row = image_lab[y, x_start_scan:x_end_scan, :]
    diff = np.abs(row - mean_bg_color)  # absolute difference per channel
    if np.any(diff > threshold):
        top_of_head = y
        break

# Determine 'y' position
selectionY = top_of_head - 25

# Adjust 'y' of it overshoots the top of the image
if selectionY < 1:
    selectionY = 3

# Face detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
    gray, scaleFactor=s_Factor, minNeighbors=m_Neighbours
)

# If no face has been detected
if len(faces) == 0:
    fail("No face found in image")

# Select the face closest to image center
image_center = (ImageW // 2, ImageH // 2)
face = min(
    faces,
    key=lambda f: (f[0] + f[2] // 2 - image_center[0]) ** 2
    + (f[1] + f[3] // 2 - image_center[1]) ** 2,
)

faceX, faceY, faceW, faceH = face

# Selection logic starts here
oppX = ImageW - (
    faceX + faceW
)  # distance from right edge of the face to the right edge of the image
lessX = min(faceX, oppX)  # smaller of the left and right 'distance to x'\

# Determine selectionX (left co-ordinate for selection)
if lessX == faceX:
    selectionX = faceX - 150 if faceX > 150 else 3
elif lessX == oppX:
    selectionX = faceX - 150 if oppX > 150 else faceX - (oppX - 3)
else:
    selectionX = 3  # fallback (should rarely trigger)

# Initial width and height based on Aspect Ratio
selectionW = faceW + ((faceX - selectionX) * 2)
selectionH = selectionW * targeted_Height_AR

# Adjust widht if height overshoots image bottom
if selectionH + selectionY >= ImageH:
    selectionH = ImageH - (selectionY + 3)

    # Recalculated width to maintain Aspect Ratio of height is adjusted
    recalculated_selectionW = selectionH * targeted_Width_AR
    width_Diff = selectionW - recalculated_selectionW
    eachSideAdjustable = width_Diff / 2
    selectionX = selectionX + eachSideAdjustable
    selectionW = recalculated_selectionW

# Clamp and Round off values
selectionX = max(0, selectionX)
selectionY = max(0, selectionY)

if selectionX + selectionW > ImageW:
    selectionW = ImageW - selectionX

if selectionY + selectionH > ImageH:
    selectionH = ImageH - selectionY

selectionX = int(round(selectionX))
selectionY = int(round(selectionY))
selectionW = int(round(selectionW))
selectionH = int(round(selectionH))

print(f"{selectionX}, {selectionY}, {selectionW}, {selectionH}")

cv2.destroyAllWindows()
