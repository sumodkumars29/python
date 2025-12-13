# border_detector.py

import cv2
from pathlib import Path
import numpy as np

from typing import Optional, Tuple

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class BorderDetector:
    def __init__(self, image_path: str, debug: bool = False):
        self.image_path = Path(image_path)
        self.debug = debug
        self.original = None
        self.gray = None
        self.edges = None
        self.corners = None
        self.border_box = None  # (x1, y1, x2, y2) or four points

    # ------------------------------
    # LOAD
    # ------------------------------
    def load(self):
        self.original = cv2.imread(str(self.image_path))
        if self.original is None:
            raise ValueError(f"Failed to load image: {self.image_path}")
        return self

    # ------------------------------
    # PREPROCESS
    # ------------------------------
    def preprocess(self):
        """
        Create:
            - self.gray
            - self.norm
            - self.blur
        """

        # Convert to gray
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)

        # Histogram normalization (light/color variations across images)
        self.norm = cv2.equalizeHist(self.gray)

        # Slight blur to remove micro-noise without damaging edges
        self.blur = cv2.GaussianBlur(self.norm, (5, 5), 0)

        # Invert blurred image (simple bitwise invert)
        self.invert = cv2.bitwise_not(self.blur)

        # Invert normalized (sharper before blur)
        self.invert_norm = cv2.bitwise_not(self.norm)

        # blur inverted normalized image
        self.invert_norm_blur = cv2.GaussianBlur(self.invert_norm, (5, 5), 0)

        # Linear invert using LUT (optional, stronger inversion)
        lut = np.array([255 - i for i in range(256)], dtype=np.uint8)
        self.linear_invert = cv2.LUT(self.blur, lut)

        # Linear invert normalized image
        self.norm_linear_invert = cv2.LUT(self.norm, lut)

        # Blur linear inverted normalized image
        self.norm_linear_invert_blur = cv2.GaussianBlur(
            self.norm_linear_invert, (5, 5), 0
        )

        # Adaptive threshold
        self.thresh = cv2.adaptiveThreshold(
            self.invert_norm_blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            35,
            5,
        )

        if self.debug:
            self.show_small("Gray", self.gray)
            self.show_small("Normalized", self.norm)
            self.show_small("Blurred", self.blur)
            self.show_small("Blurred_Inverted", self.invert)
            self.show_small("Normalized_Blurred_LinearInverted", self.linear_invert)
            self.show_small("Normalized_Inverted", self.invert_norm)
            self.show_small("Normalized_Inverted_Blurred", self.invert_norm_blur)
            self.show_small("Normalized_LinearInverted", self.norm_linear_invert)
            self.show_small(
                "Normalized_LinearInverted_Blurred", self.norm_linear_invert_blur
            )
            self.show_small("Adaptive Threshold", self.thresh)

    # ------------------------------
    # RESIZE IMSHOW() IMAGE SIZE
    # ------------------------------
    def show_small(self, name, img, max_h=800):
        h, w = img.shape[:2]
        scale = max_h / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imshow(name, small)
        cv2.waitKey(0)

    # ------------------------------
    # FACE DETECTION AND IDENTIFYING BEST FACE CANDIDATE
    # ------------------------------
    def _choose_best_face(
        self, faces: np.ndarray, img_w: int, img_h: int
    ) -> Tuple[int, int, int, int]:
        """Pick face closest to image center. faces from detectMultiScale."""
        image_center = (img_w // 2, img_h // 2)
        best = min(
            faces,
            key=lambda f: (f[0] + f[2] // 2 - image_center[0]) ** 2
            + (f[1] + f[3] // 2 - image_center[1]) ** 2,
        )
        return tuple(map(int, best))

    def detect_face(
        self,
        scaleFactor: float = 1.1,
        minNeighbors: int = 3,
        min_size: Tuple[int, int] = (25, 25),
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect a single face in self.original and set self.face_box.
        Returns:
        (x, y, w, h) if found, otherwise None.
        Notes:
        - Uses Haar cascade by default
        - Should be called after load() -> self.original must be set.
        """
        if self.original is None:
            raise ValueError("Image not loaded. Call load() before detect_face().")

        img = self.original

        # Load cascade
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if face_cascade.empty():
            raise RuntimeError(f"Failed to load cascade at: {CASCADE_PATH}")

        # Optional pre_upscale to improve Haar detection
        if min(img.shape[0], img.shape[1]) < 900:
            img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w = img.shape[:2]

        # detectMultiScale params tuned for passport/headshots; adjust later if needed
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            if self.debug:
                print("[detect_face] No faces detected.")
            self.face_box = None
            return None

        # Choose best face (closest to image center)
        x, y, w, h = self._choose_best_face(faces, img_w, img_h)
        self.face_box = (x, y, w, h)

        if self.debug:
            print(f"[detect_face] Face detected: x={x}, y={y}, w={w}, h={h}")
            # show face region scaled-down for debug
            face_roi = img[y : y + h, x : x + w].copy()
            self.show_small("Face ROI", face_roi)

        img_gh, img_gw = gray.shape[:2]
        cx = x + w // 2
        cy = y + h // 2

        left = x
        right = img_gw - (x + w)
        top = y
        bottom = img_gh - (y + h)

        print("[face_metrics]")
        print("image:", img_gh, img_gw)
        print("face_box", x, y, w, h)
        print("center:", cx, cy)
        print("left:", left, "right:", right, "top:", top, "bottom:", bottom)

        return self.face_box

    # ------------------------------
    # BORDER DETECTION
    # ------------------------------
    def detect_borders(self):
        """To be filled later — detect top/right/bottom/left borders."""
        return self

    # ------------------------------
    # CORNER DETECTION
    # ------------------------------
    def detect_corners(self):
        """To be filled later — derive the 4 corner coordinates."""
        return self

    # ------------------------------
    # WARP USING CORNERS
    # ------------------------------
    def warp(self):
        """To be filled later — warp using detected corners."""
        return None

    # ------------------------------
    # ENTRY POINT
    # ------------------------------
    def process(self):
        """High-level call chain — returns warped image & diagnostic data."""
        self.load()
        self.preprocess()
        self.detect_borders()
        self.detect_corners()
        warped = self.warp()

        return {
            "warped": warped,
            "corners": self.corners,
            "border_box": self.border_box,
        }
