import cv2
import mediapipe as mp

mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Load an image
img = cv2.imread(r"C:\Users\S10DIGITAL\python\py_GIMP\openCV\Pic3.jpg")

with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie:
    result = selfie.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    mask = result.segmentation_mask

# Convert mask to visible form
mask_vis = (mask * 255).astype("uint8")

cv2.imwrite(r"C:\Users\S10DIGITAL\python\py_GIMP\openCV\mask_output.png", mask_vis)
print("Saved mask_output.png")

