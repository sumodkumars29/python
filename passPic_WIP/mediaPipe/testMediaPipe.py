
import cv2
import mediapipe as mp

mp_selfie = mp.solutions.selfie_segmentation

img = cv2.imread(r"C:\Users\S10DIGITAL\python\py_GIMP\openCV\Pic14.jpeg")  # replace with your test image
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

with mp_selfie.SelfieSegmentation(model_selection=1) as segment:
    res = segment.process(rgb)

mask_mp = res.segmentation_mask  # float mask, same size as input

print("Mask shape:", mask_mp.shape)
print("Mask min/max:", mask_mp.min(), mask_mp.max())

cv2.imshow("MediaPipe Mask", mask_mp)
cv2.waitKey(0)
