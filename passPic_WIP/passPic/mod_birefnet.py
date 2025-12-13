import torch
import cv2
import numpy as np
import sys
from pathlib import Path

BIREFNET_ROOT = Path(__file__).resolve().parents[1] / "BiRefNet"
sys.path.insert(0, str(BIREFNET_ROOT))

from models.birefnet import BiRefNet

# from typing import Tuple


class BiRefNetWrapper:
    def __init__(self, weights_path: str, device: str = "cpu"):
        """
        Load BiRefNet model with weights.
        """
        self.device = device
        self.model = BiRefNet().to(device)
        state = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()

    # ------------------------------------------------------------
    #   MASK GENERATION
    # ------------------------------------------------------------
    def get_mask(self, image: np.ndarray, safe_size: int = 992) -> tuple:
        """Generate a foreground mask and bounding box from input image.

        Args:
            image(np.ndarray): Input BGR image.
            safe_size (int): Resize input to this square size (multiple of 31).

        Returns:
            mask (np.ndarray): Final eroded mask 0-255, resized to original image size.
            mask_box (tuple): Bounding box (x1, y1, x2, y2)."""

        if image is None:
            raise ValueError("Input image is None")

        orig_h, orig_w = image.shape[:2]

        # Convert BGR -> RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to safe square
        img_resized = cv2.resize(img_rgb, (safe_size, safe_size))

        # Prepare tensor
        img_t = (
            torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        )
        img_t = img_t.to(self.device)

        with torch.no_grad():
            preds = self.model(img_t)
            pred = preds[-1][0].cpu().numpy().squeeze(0)

        # Sigmoid
        pred = 1 / (1 + np.exp(-pred))
        pred = np.clip(pred, 0, 1)

        # RESIZE BACK TO ORIGINAL
        pred_resized = cv2.resize(
            pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )
        mask = (pred_resized * 255).astype(np.uint8)

        # Erode mask for tighter borders
        kernel = np.ones((3, 3), np.uint8)
        mask_eroded = cv2.erode(mask, kernel, iterations=1)

        # Bounding box of mask
        ys, xs = np.where(mask_eroded > 0)
        if len(xs) == 0 or len(ys) == 0:
            mask_box = (0, 0, orig_w, orig_h)
        else:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            mask_box = (x1, y1, x2, y2)

        return mask_eroded, mask_box

    # ----------------------------------------------------------------------
    #   WARPING (IMAGE + MASK USE SAME FUNCTION)
    # ----------------------------------------------------------------------
    @staticmethod
    def warp_with_matrix(
        image: np.ndarray, warp_matrix: np.ndarray, output_size: tuple
    ) -> np.ndarray:
        """
        Warp an image or mask using the same perspective matrix.
        Args:
            image (np.ndarray): Image or mask.
            warp_matrix (np.ndarray): 3x3 perspective transform matrix.
            output_size (tuple): (width, height) of the output canvas.

        Returns:
            warped (np.ndarray): Warped image or mask.
        """
        if warp_matrix.shape != (3, 3):
            raise ValueError("warp_matrix must be 3x3")

        warped = cv2.warpPerspective(
            image,
            warp_matrix,
            output_size,  # (width, height)
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return warped

    # ----------------------------------------------------------------------
    #   BACKGROUND REPLACEMENT
    # ----------------------------------------------------------------------
    @staticmethod
    def apply_mask(
        image: np.ndarray, mask: np.ndarray, new_background: np.ndarray
    ) -> np.ndarray:
        """
        Replace background using mask.
        Args:
            image (np.ndarray): BGR foreground image.
            mask (np.ndarray): 0-255 mask.
            new_background (np.ndarray): BGR background.

        Returns:
            output_img (np.ndarray): Final composited image.
        """
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError("Image and mask must have same height & widht.")
        if image.shape != new_background.shape:
            raise ValueError("Image and new_background must be identical size.")

        alpha = mask.astype(float) / 255.0

        output_img = (
            image * alpha[..., None] + new_background * (1 - alpha[..., None])
        ).astype(np.uint8)

        return output_img
