import os

import cv2

# Path to the folder containing imaages to target
input_folder = "/home/sumodk/Desktop/Pictures"
output_folder = "/home/sumodk/Desktop/Pictures/results"

os.makedirs(output_folder, exist_ok=True)

# Load Haar cascade for face detection (comes with OpenCV)
face_cascade = cv2.CascadeClassifier(
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
)

# Detection parameters (adjustable)
s_Factor = 1.2  # Smaller -> more sensitive
m_Neighbours = 6  # Larger -> stricter
t_Ratio = 815 / 1063  # width / height

# Loop through the images in the folder
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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=s_Factor, minNeighbors=m_Neighbours
    )

    if len(faces) == 0:
        print(f"No face detected in {filename}")
        continue

    img_h, img_w = img.shape[:2]
    img_center = (img_w / 2, img_h / 2)

    # --- Step 1 : Select face closest to image center ---
    def face_center(face):
        x, y, w, h = face
        return (x + w / 2, y + h / 2)

    def distance_to_center(face):
        fx, fy = face_center(face)
        return (fx - img_center[0]) ** 2 + (fy - img_center[1]) ** 2

    closest_face = min(faces, key=distance_to_center)
    x, y, w, h = closest_face

    # -- Step 2 : Calculate crop rectangle maintaining aspect ratio --
    top_pad = min(int(h * 0.25), y)  # up to 25% of face height above head

    crop_w = w
    crop_h = int(crop_w / t_Ratio)

    if crop_h < h + top_pad:
        crop_h = h + top_pad
        crop_w = int(crop_h * t_Ratio)

    # Center face horizontally
    x1 = max(int(x + w / 2 - crop_w / 2), 0)
    x2 = min(x1 + crop_w, img_w)

    # Align vertically (include top of head of possible)
    y1 = max(int(y - top_pad), 0)
    y2 = min(y1 + crop_h, img_h)

    # Adjust of hitting image edge
    if x2 > img_w:
        x1 = max(img_w - crop_w, 0)
        x2 = img_w
    if y2 > img_h:
        y1 = max(img_h - crop_h, 0)
        y2 = img_h

    # --- Step 3: Draw rectangles ---
    # Green : face
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # Red : proposed crop area
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # Save the image automatically
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, img)

print("All images prcessed and saved to ", output_folder)

# cv2.imshow("Face Detection", img)
# # cv2.waitKey(0)
# while True:
#     key = cv2.waitKey(100)
#     if key != -1:
#         break

cv2.destroyAllWindows()
