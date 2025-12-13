import cv2

from mod_birefnet import BiRefNetWrapper

# ===================== CONFIG =====================
IMAGE_PATH = (
    r"C:\Users\S10DIGITAL\python\py_GIMP\openCV\trial_directory\Pic7_1100h.jpeg"
)
WEIGHTS_PATH = r"C:\Users\S10DIGITAL\python\py_GIMP\BiRefNet\weights\BiRefNet-general-epoch_244.pth"

# ===================== LOAD IMAGE =====================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise RuntimeError("Failed to load test image")

print("Image loaded:", img.shape)

# ===================== LOAD MODEL =====================
biref = BiRefNetWrapper(WEIGHTS_PATH, device="cpu")
print("BiRefNet loaded")

# ===================== RUN MASK =====================
mask, mask_box = biref.get_mask(img)

print("Mask shape:", mask.shape)
print("Mask dtype:", mask.dtype)
print("Mask box:", mask_box)

# ===================== BASIC VALIDATION =====================
assert mask.shape[:2] == img.shape[:2], "Mask size mismatch"
assert mask_box[2] > mask_box[0], "Invalid mask box X"
assert mask_box[3] > mask_box[1], "Invalid mask box Y"

print("✅ BiRefNet module test PASSED")
