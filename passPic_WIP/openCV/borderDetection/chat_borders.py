import cv2
import numpy as np
import sys

# import os
# import matplotlib.pyplot as plt


def order_quad(pts):
    # order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(img, pts):
    rect = order_quad(pts)
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    maxWidth = max(int(width_a), int(width_b))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped, M, rect


def find_outer_quad(image, debug=False):
    # resize for speed but keep scale
    orig_h, orig_w = image.shape[:2]
    scale = 800.0 / max(orig_w, orig_h) if max(orig_w, orig_h) > 800 else 1.0
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    # close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1000:  # skip small
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            candidates.append((area, approx.reshape(4, 2)))
    if not candidates:
        return None, scale
    # pick largest-area quadrilateral
    quad = max(candidates, key=lambda x: x[0])[1]
    # scale quad back to original image size
    quad = quad / scale
    return quad, scale


def shrink_mask_by_pixels(binary_mask, shrink_px):
    # binary_mask: 0/255 mask of outer area (uint8)
    # compute distance transform from background (so edges inside become small dist)
    inv = cv2.bitwise_not(binary_mask)
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    inner = (dt > shrink_px).astype("uint8") * 255
    return inner


def main(input_path, shrink_pixels=10, debug=False):
    img = cv2.imread(input_path)
    if img is None:
        print("Failed to load:", input_path)
        return
    # 1) find outer quad in photo
    quad, scale = find_outer_quad(img, debug=debug)
    if quad is None:
        print(
            "No outer rectangle found. Try adjusting thresholds or use 'Hough Lines'."
        )
        return
    # 2) Warp to get flat image
    warped, M, ordered = four_point_transform(img, quad)
    # 3) Create a mask of the whole warped image polygon (just fill full rectangle)
    h, w = warped.shape[:2]
    outer_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(
        outer_mask,
        np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32),
        255,
    )
    # 4) Compute inner area by shrinking mask (distance tranform)
    inner_mask = shrink_mask_by_pixels(outer_mask, shrink_pixels)
    # Optionally : detect the border region by finding strong edges and then shrink
    # Save output
    cv2.imwrite("warped.jpg", warped)
    cv2.imwrite("inner_mask.png", inner_mask)
    # overlay for visualization
    vis = warped.copy()
    contours, _ = cv2.findContours(
        inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    cv2.imwrite("warped_with_inner_overlay.jpg", vis)
    print("Saved: wraped.jpg, inner_mask.png, warped_with_inner_overlay.jpg")
    print(
        "Inner reqion is inner_mask.png(white=usable area). shrink_pixels=",
        shrink_pixels,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python borders.py input.jpg [shrink_px]")
    else:
        inp = sys.argv[1]
        shrink = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        main(inp, shrink_pixels=shrink)
