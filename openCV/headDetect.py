import os

import cv2
import numpy as np

# Path to the folder containing imaages to target
input_folder = "/home/sumodk/Desktop/Pictures"
output_folder = "/home/sumodk/Desktop/Pictures/results"

os.makedirs(output_folder, exist_ok=True)
for filename in os.listdir(input_folder):
    if not (
        filename.lower().endswith(".jpg")
        or filename.lower().endswith(".png")
        or filename.lower().endswith(".jpeg")
    ):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping unreadable image: {filename}")
        continue
    # Load Image
    img = cv2.imread(img_path)
    H, W, _ = img.shape

    # Convert to LAB colour space for better colour difference handling
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Step 1: Sample background (rows 5-80, first 150 colunns)
    sample_region = img_lab[5:80, 0:150]
    mean_bg_color = sample_region.mean(axis=(0, 1))

    # Step 2 : Scan from top downward
    threshold = 15  # LAB os more sensitive; lower threshold
    top_of_head = 0

    # Optional: scan only the central 80% columns to avoid edge noise
    x_start_scan = int(W * 0.1)
    x_end_scan = int(W * 0.9)

    for y in range(H):
        row = img_lab[y, x_start_scan:x_end_scan, :]
        diff = np.abs(row - mean_bg_color)  # absolute difference per channel
        if np.any(diff > threshold):
            top_of_head = y
            break

    print(f"Top of head for {filename} detected at row: ", top_of_head)

    # Step 3: Determine line position
    line_y = top_of_head - 25
    line_thickness = 2

    # Step 4: Adjust if line goes above image
    if line_y < 0:
        line_y = 3
        line_thickness = 1

    # Step 5: Draw horizontal line (200 px wide, centered)
    line_length = 200
    x_start = max(0, (W - line_length) // 2)
    x_end = x_start + line_length

    cv2.line(img, (x_start, line_y), (x_end, line_y), (0, 0, 255), line_thickness)

    # Save the image automatically
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, img)

    cv2.destroyAllWindows()
