from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Union


class PassportBorderDetector:
    # Comprehensive border detector for passport photos handling all scenarios.

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.results = {}
        self.image = None
        self.gray = None

    def load_image(self, image_path: Union[str, Path]) -> bool:
        # Load and prepare image for processing.
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray.shape

        if self.debug:
            print(f"Image loaded: {self.width}x{self.height} pixels")

        return True

    def detect_all_borders(self, image_path: Union[str, Path]) -> Dict:
        """
        Main detection pipeline - handles all scenarios
        Returns:
        Dictionary containing all detected borders,corners and confidence scores
        """
        self.load_image(image_path)

        # Initialize results structure
        self.results = {
            "original_size": (self.width, self.height),
            "scenario": None,
            "confidence": 0.0,
            "outer_border": None,  # Bleed/border area
            "inner_border": None,  # Line border
            "image_area": None,  # Actual photo area
            "corners": None,  # Four corners of image area
            "is_skewed": False,
            "skew_angle": 0.0,
            "has_frame": False,  # Outer frame from photograph
        }

        # Step1: Detect if there's an outer frame
        frame_detected = self.detect_outer_frame()
        self.results["has_frame"] = frame_detected

        # Step2: Identify scenario and detect borders
        scenario = self.identify_scenario()
        self.results["scenario"] = scenario

        if scenario in [1, 2, 5]:
            # Has borders = detect them
            borders_found = self.detect_solid_borders()
            if borders_found and scenario == 2:
                # Also detect line border inside
                line_border = self.detect_line_border()
                self.results["inner_border"] = line_border
        else:
            # Scenario 3 or 4 - no borders, detect subject
            self.detect_subject_no_border()

        # Step3: Refine corners and calculate skew
        self.refine_corners_and_skew()

        # Step4: Calculate confidence
        self.calculate_confidence()

        return self.results

    def identify_scenario(self) -> int:
        """
        Idendify which scenario we're dealing with
        Returns scenario number (1-5)
        """
        # Analyze edges around perimeter
        perimeter_edges = self.analyze_perimeter()

        # Check for solid border (high uniform colour around edges)
        has_solid_border = self.check_for_solid_border()

        # Check for line border
        has_line_border = self.check_for_line_border()

        # Check for complex background
        has_complex_bg = self.has_complex_background()

        # Determine scenario
        if has_solid_border and not has_line_border:
            return 1  # Solid border only
        elif has_solid_border and has_line_border:
            return 2  # Solid border with line border
        elif not has_solid_border and has_complex_bg:
            return 3  # No border, complex background
        elif not has_solid_border and not has_complex_bg:
            return 4  # No border, solid background
        else:
            # Check for outer frame (photograph of photo)
            return 5

    def analyze_perimeter(self, sample_width: int = 50) -> Dict:
        """
        Analyze image perimeter to understand border characteristics.
        """
        height, width = self.gray.shape

        # Sample strips from all four sides
        top_strip = self.gray[0:sample_width, :]
        bottom_strip = self.gray[height - sample_width : height, :]
        left_strip = self.gray[:, 0:sample_width]
        right_strip = self.gray[:, width - sample_width : width]

        # Calculate statistics
        strips = {
            "top": top_strip,
            "bottom": bottom_strip,
            "left": left_strip,
            "right": right_strip,
        }

        stats_dict = {}
        for side, strip in strips.items():
            mean = np.mean(strip)
            std = np.std(strip)
            uniformity = std / mean if mean > 0 else 0

            stats_dict[side] = {
                "mean": mean,
                "std": std,
                "uniformity": uniformity,
                "has_edge": std > 20,  # High std indicates edge
            }

        return stats_dict

    def check_for_solid_border(self) -> bool:
        """
        Check if image has solid color border around edges.
        """
        # Sample corners to detect border color
        corner_size = 100
        corners = [
            self.image[0:corner_size, 0:corner_size],  # Top-left
            self.image[0:corner_size, -corner_size:],  # Top-right
            self.image[-corner_size:, 0:corner_size],  # Bottom-left
            self.image[-corner_size:, -corner_size:],  # Bottom-right
        ]

        # Check if corners have uniform color (low std)
        corner_uniformities = []
        for corner in corners:
            if corner.size > 0:
                std = np.std(corner)
                corner_uniformities.append(std)

        # Border likely if at least 3 corners are uniform
        uniform_corners = sum(1 for std in corner_uniformities if std < 15)

        return uniform_corners >= 3

    def check_for_line_border(self) -> bool:
        """
        Detect thin line borders (1-5px thick).
        """
        # Use Canny edge detection
        edges = cv2.Canny(self.gray, 50, 150)

        # Look for long straight lines near edges
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=self.width // 4,
            maxLineGap=10,
        )

        if lines is not None:
            # Check if lines form a rectangle near image boundaries
            border_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is near edge
                if (
                    x1 < 50
                    or x2 < 50
                    or x1 > self.width - 50
                    or x2 > self.width - 50
                    or y1 < 50
                    or y2 < 50
                    or y1 > self.height - 50
                    or y2 > self.height - 50
                ):
                    border_lines += 1

            return border_lines >= 2

        return False

    def has_complex_background(self) -> bool:
        """
        Determine if background is complex (not uniform).
        """
        # Focus on perimeter areas (likely background in passport photos)
        margin = 100
        bg_area = self.gray[margin : self.height - margin, margin : self.width - margin]

        if bg_area.size == 0:
            return False

        # Calculate texture complexity
        sobelx = cv2.Sobel(bg_area, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(bg_area, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)

        complexity = np.mean(gradient_mag)

        # Complex if gradient_mag is high
        return complexity > 10

    def detect_outer_frame(self) -> bool:
        """
        Detect if image has an outer frame (photograph of a printed photo).
        Usually shows shadows, perspective distortion, etc.
        """
        # Check for vignetting or shadows at edges
        edges = cv2.Canny(self.gray, 50, 150)

        # Count edges in outer 10% of image
        border_region = 0.1
        border_mask = np.zeros_like(self.gray)
        border_mask[: int(self.height * border_region), :] = 1
        border_mask[-int(self.height * border_region) :, :] = 1
        border_mask[:, : int(self.width * border_region)] = 1
        border_mask[:, -int(self.width * border_region) :] = 1

        border_edges = edges[border_mask == 1]
        edge_density = np.sum(border_edges > 0) / border_edges.size

        # Also check colour saturation at edges (photographs often have colour casts)
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        border_saturation = saturation[border_mask == 1]
        mean_border_sat = np.mean(border_saturation)

        # Frame likely if high edge density and abnormal saturation at borders
        return edge_density > 0.1 and mean_border_sat > 50

    def detect_solid_borders(self) -> Dict:
        """
        Detect solid color borders (for scenarios 1, 2, 5).
        Returns border dimensions on all four sides.
        """
        height, width = self.gray.shape

        # Use scanning approach from all four sides
        border_top = self.scan_from_side("top")
        border_bottom = self.scan_from_side("bottom")
        border_left = self.scan_from_side("left")
        border_right = self.scan_from_side("right")

        borders = {
            "top": border_top,
            "bottom": border_bottom,
            "left": border_left,
            "right": border_right,
        }

        # Calculate image area inside borders
        image_top = border_top if border_top > 0 else 0
        image_bottom = height - border_bottom if border_bottom > 0 else height
        image_left = border_left if border_left > 0 else 0
        image_right = width - border_right if border_right > 0 else width

        self.results["outer_border"] = borders
        self.results["image_area"] = (image_left, image_top, image_right, image_bottom)

        # Initial corner estimation
        self.results["corners"] = [
            (image_left, image_top),  # Top-left
            (image_right, image_top),  # Top-right
            (image_right, image_bottom),  # Bottom-right
            (image_left, image_bottom),  # Bottom-left
        ]

        if self.debug:
            print(f"Detected borders: {borders}")
            print(f"Image area: {self.results['image_area']}")

        return borders

    def scan_from_side(self, side: str, max_scan: int = 200) -> int:
        """
        Scan from one side to detect border width.
        Returns border width in pixels.
        """
        height, width = self.gray.shape
        scan_distance = min(max_scan, min(height, width) // 4)

        if side == "top":
            # Scan from top downward
            scan_region = self.gray[0:scan_distance, :]
            base_sample = self.gray[0, :]
        elif side == "bottom":
            # Scan from bottom upward
            scan_region = self.gray[height - scan_distance : height, :]
            base_sample = self.gray[height - 1, :]
        elif side == "left":
            # Scan from left to right
            scan_region = self.gray[:, 0:scan_distance]
            base_sample = self.gray[:, 0]
        elif side == "right":
            # Scan from right to left
            scan_region = self.gray[:, width - scan_distance : width]
            base_sample = self.gray[:, width - 1]
        else:
            return 0

        # Calculate colour differences
        border_width = 0
        threshold = 20  # Colour difference threshold

        for i in range(1, scan_distance):
            if side in ["top", "bottom"]:
                current_row = (
                    scan_region[i, :] if side == "top" else scan_region[-i - 1, :]
                )
                diff = np.mean(
                    np.abs(current_row.astype(float) - base_sample.astype(float))
                )
            else:  # left or right
                current_col = (
                    scan_region[:, i] if side == "left" else scan_region[:, -i - 1]
                )
                diff = np.mean(
                    np.abs(current_col.astype(float) - base_sample.astype(float))
                )

            if diff > threshold:
                border_width = i
                break

        # If no border detected, check if entire scan region is uniform
        if border_width == 0:
            if side in ["top", "bottom"]:
                uniformity = np.std(scan_region)
            else:
                uniformity = np.std(scan_region)

            # If very uniform, might be a large border
            if uniformity < 10:
                border_width = scan_distance

        return border_width

    def detect_line_border(self) -> Dict:
        """
        Detect thin line borders inside solid border.
        """
        # Use edge detection focused on the inner area (inside solid border)
        if self.results["outer_border"]:
            top, bottom, left, right = (
                self.results["outer_border"]["top"],
                self.results["outer_border"]["bottom"],
                self.results["outer_border"]["left"],
                self.results["outer_border"]["right"],
            )

            # Region inside outer_border
            inner_region = self.gray[
                top : self.height - bottom, left : self.width - right
            ]

        else:
            inner_region = self.gray

        if inner_region.size == 0:
            return None

        # Detect edges
        edges = cv2.Canny(inner_region, 50, 150)

        # Find lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=min(inner_region.shape) // 3,
            maxLineGap=10,
        )

        line_border = {
            "top": None,
            "bottom": None,
            "left": None,
            "right": None,
            "lines": [],
        }

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Adjust co-ordinates to original image
                if self.results["outer_border"]:
                    x1 += left
                    x2 += left
                    y1 += top
                    y2 += top

                line_border["lines"].append((x1, y1, x2, y2))

        return line_border

    def detect_subject_no_border(self):
        """
        Detect subject in images without borders
        """
        # For solid background
        if self.results["scenario"] == 4:
            self.detect_subject_solid_bg()
        else:
            # Complex background
            self.detect_subject_complex_bg()

    def detect_subject_solid_bg(self):
        """
        Detect subject against solid background.
        Uses thresholding and contour detection.
        """
        # Apply adaptive thresholding
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)

        # Try different thresholding methods
        _, thresh_otsu = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            # Fallback to edge_based detection
            self.detect_subject_edges()
            return

        # Find largest contour (likely the subject)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add some margin
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(self.width - x, w + 2 * margin)
        h = min(self.height - y, h + 2 * margin)

        self.results["image_area"] = (x, y, x + w, y + h)
        self.results["corners"] = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

    def detect_subject_complex_bg(self):
        """
        Detect subject against complex background.
        Uses advanced segmentation techniques.
        """
        # Try GrabCut for complex backgrounds
        mask = np.zeros(self.gray.shape[:2], np.uint8)

        # Define ROI (assume subject is roughly centered)
        roi_width = self.width // 2
        roi_height = self.height // 2
        roi_x = (self.width - roi_width) // 2
        roi_y = (self.height - roi_height) // 2

        # Initialize mask for GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        rect = (roi_x, roi_y, roi_width, roi_height)

        try:
            cv2.grabCut(
                self.image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT
            )

            # Create mask where sure/fgd are 1, rest are 0
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

            # Find contours in mask
            contours, _ = cv2.findContours(
                mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                self.results["image_area"] = (x, y, x + w, y + h)
                self.results["corners"] = [
                    (x, y),
                    (x + w, y),
                    (x + w, y + h),
                    (x, y + h),
                ]
                return
        except:
            pass

        # Fallback to edge-based detection
        self.detect_subject_edges()

    def detect_subject_edges(self):
        """
        Fallback method using edge detection
        """
        edges = cv2.Canny(self.gray, 50, 150)

        # Dilate edges to connect gaps
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Combine all contours
            all_points = np.vstack([contour for contour in contours])
            x, y, w, h = cv2.boundingRect(all_points)

            self.results["image_area"] = (x, y, x + w, y + h)
            self.results["corners"] = [
                (x, y),
                (x + w, y),
                (x + w, y + h),
                (x, y + h),
            ]
        else:
            # Default to entire image
            self.results["image_area"] = (0, 0, self.width, self.height)
            self.results["corners"] = [
                (0, 0),
                (self.width, 0),
                (self.width, self.height),
                (0, self.height),
            ]

    def refine_corners_and_skew(self):
        """
        Refine corner detection and calculate skew angle.
        """
        if not self.results["corners"]:
            return

        corners = self.results["corners"]

        # Convert to numpy array
        corners_np = np.array(corners, dtype=np.float32)

        # Find best fitting quadrilateral
        if len(corners_np) >= 4:
            # Order corners: top-left, top-right, bottom-right, bottom-left
            ordered_corners = self.order_corners(corners_np)

            # Calculate skew angle from horizontal lines
            top_line_angle = np.arctan2(
                ordered_corners[1][1] - ordered_corners[0][1],
                ordered_corners[1][0] - ordered_corners[0][0],
            ) * (180 / np.pi)

            bottom_line_angle = np.arctan2(
                ordered_corners[2][1] - ordered_corners[3][1],
                ordered_corners[2][0] - ordered_corners[3][0],
            ) * (180 / np.pi)

            avg_skew = (top_line_angle + bottom_line_angle) / 2

            # Check if skewed (angles not close to 0)
            self.results["is_skewed"] = abs(avg_skew) > 0.5
            self.results["skew_angle"] = avg_skew

            self.results["corners"] = ordered_corners.tolist()

            if self.debug and self.results["is_skewed"]:
                print(f"Image is skewed by {avg_skew:.2f} degrees")

    def order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners in consistent order : TL, TR, BR and BL.
        """
        # Find centroid
        centroid = np.mean(corners, axis=0)

        # Sort by angle from centroid
        angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])

        # Sort corners by angle
        sorted_indices = np.argsort(angles)
        sorted_corners = corners[sorted_indices]

        # Ensure we start with top-left (smallest x+y)
        distances = np.sum(sorted_corners, axis=1)
        start_index = np.argmin(distances)

        ordered = np.roll(sorted_corners, -start_index, axis=0)

        return ordered

    def calculate_confidence(self):
        """
        Calculate confidence score for the detection.
        """
        confidence = 1.0

        # Penalize if corners don't form a proper rectangle
        if self.results["corners"]:
            corners = np.array(self.results["corners"])

            # Check aspect ratio (passport photos are typically 2:3 or 3:4)
            width = np.linalg.norm(corners[1] - corners[0])
            height = np.linalg.norm(corners[2] - corners[1])

            if height > 0:
                aspect_ratio = width / height
                expected_ratio = 0.75  # 3:4 aspect ratio

                # Penalize if aspect ratio is far from expected
                ratio_diff = abs(aspect_ratio - expected_ratio) / expected_ratio
                confidence *= max(0, 1 - ratio_diff)

        # Penalize if detection area is too small
        if self.results["image_area"]:
            x1, y1, x2, y2 = self.results["image_area"]
            area = (x2 - x1) * (y2 - y1)
            total_area = self.width * self.height

            if area / total_area < 0.1:  # Less than 10% of image
                confidence *= 0.5

        self.results["confidence"] = min(1.0, max(0.0, confidence))

    def get_rectification_transform(self):
        """
        Get transformation matrix to rectify (deskew) the image.
        """
        if not self.results["corners"] or not self.results["is_skewed"]:
            return None

        src_corners = np.array(self.results["corners"], dtype=np.float32)

        # Calculate destination rectangle (preserve aspect ratio)
        width = int(np.linalg.norm(src_corners[1] - src_corners[0]))
        height = int(np.linalg.norm(src_corners[2] - src_corners[1]))

        dst_corners = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
        )

        # Calculate perspective transform
        transform_matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)

        return transform_matrix, (width, height)

    def rectify_image(self):
        """
        Apply rectification to deskew the image.
        Returns rectified image and mask.
        """
        transform_info = self.get_rectification_transform()

        if transform_info is None:
            return self.image, np.ones_like(self.gray)

        transform_matrix, (width, height) = transform_info

        # Apply perspective transform
        rectified = cv2.warpPerspective(self.image, transform_matrix, (width, height))

        # Create mask of valid pixels
        mask = cv2.warpPerspective(
            np.ones_like(self.gray) * 255, transform_matrix, (width, height)
        )

        return rectified, mask

    def extract_subject_area(self, rectified_image=None):
        """
        Extract just the subject area (inside borders).
        """
        if rectified_image is None:
            rectified_image = self.image

        if self.results["image_area"]:
            x1, y1, x2, y2 = self.results["image_area"]

            # Convert to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Ensure co-ordinates are within bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(rectified_image.shape[1], x2)
            y2 = min(rectified_image.shape[0], y2)

            subject = rectified_image[y1:y2, x1:x2]

            return subject

        return rectified_image

    def visualize_results(self, save_path=None):
        """
        Create visualization of detection results.
        """

        if self.image is None:
            return

        vis_image = self.image.copy()

        # Draw detected borders
        if self.results["outer_border"]:
            borders = self.results["outer_border"]
            cv2.rectangle(
                vis_image,
                (borders["left"], borders["top"]),
                (self.width - borders["right"], self.height - borders["bottom"]),
                (0, 255, 0),
                2,
            )  # Green for outer border

        # Draw line border if detected
        if self.results["inner_border"] and self.results["inner_border"]["lines"]:
            for line in self.results["inner_border"]["lines"]:
                x1, y1, x2, y2 = line
                cv2.line(
                    vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2
                )  # Blue for line border

        # Draw corners
        if self.results["corners"]:
            for i, (x, y) in enumerate(self.results["corners"]):
                color = (0, 165, 255)  # Orange
                cv2.circle(vis_image, (int(x), int(y)), 10, color, -1)
                cv2.putText(
                    vis_image,
                    str(i + 1),
                    (int(x) + 15, int(y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

        # Draw image area
        if self.results["image_area"]:
            x1, y1, x2, y2 = self.results["image_area"]
            cv2.rectangle(
                vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2
            )  # Red for image area

        # Add text overlay
        text_y = 30
        cv2.putText(
            vis_image,
            f"Scenario: {self.results['scenario']}",
            (10, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            vis_image,
            f"Confidence: {self.results['confidence']:.2f}",
            (10, text_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if save_path:
            cv2.imwrite(str(save_path), vis_image)

        return vis_image


# ===========================================================================
# MAIN EXECUTION AND USAGE EXAMPLES
# ===========================================================================


def process_passport_image(image_path, output_dir=None, debug=True):
    """
    Complete processing pipeline for passport images.
    Args:
    image_path: Path to input image
    output_dir: Directory to save results (optional)
    debug: Enable debug outputs

    Returns:
        Dictionary with all processing results
    """

    detector = PassportBorderDetector(debug=debug)

    # Step1: Detect all borders and analyze image
    results = detector.detect_all_borders(image_path)

    # Step2: Rectify if skewed
    if results["is_skewed"]:
        rectified, mask = detector.rectify_image()
    else:
        rectified = detector.image.copy()
        mask = None

    # Step3: Extract subject area
    subject_area = detector.extract_subject_area(rectified)

    # Step4: Save results if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save visualization
        vis = detector.visualize_results()
        cv2.imwrite(str(output_dir / "detection_visualization.jpg"), vis)

        # Save rectified_image
        if results["is_skewed"]:
            cv2.imwrite(str(output_dir / "rectified_image.jpg"), rectified)
            cv2.imwrite(str(output_dir / "rectification_mask.jpg"), mask)

        # Save subject area
        cv2.imwrite(str(output_dir / "subject_area.jpg"), subject_area)

        # Save results as JSON
        import json

        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif hasattr(value, "tolist"):  # For numpy types
                json_results[key] = value.tolist()
            else:
                json_results[key] = value

        with open(output_dir / "detection_results.json", "w") as f:
            json.dump(json_results, f, indent=2)

        # Save as Excel
        df = pd.DataFrame([json_results])
        df.to_excel(output_dir / "detection_results.xlsx", index=False)

    return {
        "detector": detector,
        "analysis": results,
        "rectified": rectified,
        "subject": subject_area,
        "mask": mask,
    }


def batch_process_images(image_folder, output_base_dir="processed_results"):
    """
    Process all images in a folder.
    """
    image_folder = Path(image_folder)
    output_base_dir = Path(output_base_dir)

    all_results = []

    # Find all image files
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_folder.glob(f"*{ext}"))
        image_files.extend(image_folder.glob(f"*{ext.upper()}"))

    print(f"Found {len(image_files)} images to process")

    for i, image_path in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {image_path.name}")

        try:
            # Create output directory for this image
            output_dir = output_base_dir / image_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)

            # Process the image
            result = process_passport_image(
                image_path, output_dir=output_dir, debug=True
            )

            all_results.append(
                {
                    "image": image_path.name,
                    "scenario": result["analysis"]["scenario"],
                    "confidence": result["analysis"]["confidence"],
                    "skewed": result["analysis"]["is_skewed"],
                    "skew_angle": result["analysis"]["skew_angle"],
                }
            )

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    # Create summary report
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_excel(output_base_dir / "processing_summary.xlsx", index=False)

        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(summary_df)

        # Statistics
        print(f"\nTotal processed: {len(all_results)} images")
        print(f"Average confidence: {summary_df['confidence'].mean():.2f}")
        print(f"Skewed images: {summary_df['skewed'].sum()}")

    return all_results


# =================================================================
# EXAMPLE USAGE
# =================================================================

# if __name__ == "__main__":
#     # Single image processing
#     result = process_passport_image(
#         r"C:\Users\S10DIGITAL\py_GIMP\openCV\trial_Directory\Pic1.jpg",
#         output_dir="output_results",
#         debug=True,
#     )
#
#     print("\n" + "=" * 50)
#     print("PROCESSING SUMMARY")
#     print("=" * 50)
#     for key, value in result["analysis"].items():
#         if key not in [
#             "outer_border",
#             "inner_border",
#         ]:  # Skip large dicts for readability
#             print(f"{key}: {value}")
#
#     # Display images
#     cv2.imshow("Original with Detection", result["detector"].visualize_results())
#     cv2.imshow("Subject Area", result["subject"])
#
#     if result["analysis"]["is_skewed"]:
#         cv2.imshow("Rectified image", result["rectified"])
#
#     cv2.waitKey(5)
#     cv2.destroyAllWindows()
#
# # For batch processing
# # batch_process_images("input_folder", "output_results")
