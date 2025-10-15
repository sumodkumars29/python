import os

import cv2
import numpy as np

# Path to the folder containing umages to target
input_folder = "/home/sumodk/Desktop/Pictures"
output_folder = "/home/sumodk/Desktop/Pictures/results"

# If the output_folder does not exist, create it
os.makedirs(output_folder, exist_ok=True)

# Detection parameters
s_Factor = 1.2  # smaller -> more sensitive
m_Neighbours = 6  # larger -> stricter

# Aspect ration constants (815 x 1063)
targeted_Height_AR = 1.303  # height to width (1063 / 815)
targeted_Width_AR = 0.767  # width to height (815 / 1063)

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
)

# Verify cascade file exists
if face_cascade.empty():
    raise IOError("Cannot load cascade classifier from the specified path")

# Process each image
for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping unreadable image: {filename}")
        continue

    ImageH, ImageW, _ = image.shape
    print(
        f"{filename} dimensions:\n\t Image Height = {ImageH}\n\t Image Width = {ImageW}"
    )

    # Convert to LAB colour space for better colour difference handling
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Step 1: Sample background (rows 5-80, first 150 columns)
    sample_region = image_lab[5:80, 0:150]
    mean_bg_color = sample_region.mean(axis=(0, 1))

    # Step 2: Scan from top downward
    threshold = 15  # LAB is more sensitive; lower threshold
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

    # Step 3: Determine line position
    selectionY = top_of_head - 25

    # Step 4: Adjust of line goes above image
    if selectionY < 0:
        selectionY = 3

    # Face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=s_Factor, minNeighbors=m_Neighbours
    )

    if len(faces) == 0:
        print(f"No face found in {filename}")
        continue

    # Select the face closest to image center
    image_center = (ImageW // 2, ImageH // 2)
    face = min(
        faces,
        key=lambda f: (f[0] + f[2] // 2 - image_center[0]) ** 2
        + (f[1] + f[3] // 2 - image_center[1]) * 2,
    )

    faceX, faceY, faceW, faceH = face

    # Selection logic starts
    oppX = ImageW - (
        faceX + faceW
    )  # distance from face's right edge to imnage right edge
    lessX = min(faceX, oppX)  # smaller of left and right distances

    print(
        f"{filename}:\n\t faceX={faceX},\n\t faceW={faceW},\n\t oppX={oppX},\n\t lessX={lessX}"
    )

    # determine selectionX (left coordinate for selection)
    if lessX == faceX:
        selectionX = faceX - 180 if faceX > 180 else 3
    elif lessX == oppX:
        selectionX = faceX - 180 if oppX > 180 else faceX - (oppX - 3)
    else:
        selectionX = 3  # fallback (should rarely trigger)

    # initial width and height based on AR
    selectionW = faceW + ((faceX - selectionX) * 2)
    selectionH = selectionW * targeted_Height_AR

    print(
        f"{filename}: values at first stage\n\t selectionX={selectionX},\n\t selectionY={selectionY},\n\t selectionW={selectionW},\n\t selectionH={selectionH}"
    )

    # Adjust width if height overshoots image bottom
    if selectionH + selectionY >= ImageH:
        selectionH = ImageH - (selectionY + 3)

        # recalculate width to maintain AR if height is adjusted
        recalculated_selectionW = selectionH * targeted_Width_AR
        width_Diff = selectionW - recalculated_selectionW
        eachSideAdjustable = width_Diff / 2
        selectionX = selectionX + eachSideAdjustable
        selectionW = recalculated_selectionW  # corrected width based on adjusted height

        print(
            f"{filename}: recalculate values if initial height overshoots the image bottom\n\t selectionX={selectionX},\n\t selectionW={selectionW},\n\t selectionH={selectionH}"
        )

    # clamp and round
    selectionX = int(round(selectionX))
    selectionY = int(round(selectionY))
    selectionW = int(round(selectionW))
    selectionH = int(round(selectionH))

    print(
        f"{filename}: values at final stage\n\t selectionX={selectionX},\n\t selectionY={selectionY},\n\t selectionW={selectionW},\n\t selectionH={selectionH}"
    )

    # Selection logic end

    # Draw selection rectangle

    cv2.rectangle(
        image,
        (selectionX, selectionY),
        (selectionX + selectionW, selectionY + selectionH),
        (0, 0, 255),
        1,
    )

    # Save the image
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, image)

print("All images processed and saved to ", output_folder)
cv2.destroyAllWindows()
