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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=s_Factor, minNeighbors=m_Neighbours
    )

    # Keep only closest to center
    if len(faces) > 0:
        img_center = (img.shape[1] // 2, img.shape[0] // 2)
        faces = [
            min(
                faces,
                key=lambda f: (
                    (f[0] + f[2] // 2 - img_center[0]) ** 2
                    + (f[1] + f[3] // 2 - img_center[1]) ** 2
                ),
            )
        ]

    # Draw rectangels around detected faces for testing
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # compute retract -> how much of the 'y' coordinate to retract
    # face_h = h
    # retract = int(y - (0.325 * face_h))
    #
    # if face_h > 185:
    #     if (retract - 35) <= 0:
    #         retract = 2
    #     else:
    #         retract = retract - 35
    retract = 3

    # Draw a red line at the retract position
    cv2.line(img, (x, retract), (x + w, retract), (0, 0, 255), 1)

    # Label the line at retract with coordinate value
    # cv2.putText(
    #     img,
    #     f"y={retract}",
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.6,
    #     (0, 0, 255),
    #     2,
    # )

    label = f"x={x}, y={y}, w={w}, h={h}"
    print(f"{filename}: {label}")
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
