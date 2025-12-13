import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt


class Scenario1Detector:
    """
    Detector for Scenario 1: Solid bleed borders with sharp transition to image.
    Uses Sobel gradient + Hough Lines for robust border detection.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize detector.

        Args:
            debug: If True, show intermediate processing steps
        """
        self.debug = debug
        self.image = None
        self.gray = None
        self.height = 0
        self.width = 0
        self.results = {}

    def load_image(self, image_path: str) -> bool:
        """
        Load and prepare image.

        Args:
            image_path: Path to the image file

        Returns:
            bool: True if successful, raises error if failed
        """
        # Read image using OpenCV (BGR format)
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        # Convert to grayscale for edge detection
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray.shape

        print(f"✓ Image loaded: {self.width}x{self.height} pixels")
        return True

    def apply_sobel_gradient(
        self, kernel_size: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Sobel edge detection.

        Args:
            kernel_size: Size of Sobel kernel (1, 3, 5, or 7)
                        Larger = smoother edges, more noise reduction

        Returns:
            Tuple of (gradient_x, gradient_y, gradient_magnitude)
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)

        # Calculate gradients using Sobel
        # cv2.CV_64F: 64-bit float output (preserves negative values)
        # 1, 0: Calculate derivative in x-direction (vertical edges)
        gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=kernel_size)

        # 0, 1: Calculate derivative in y-direction (horizontal edges)
        gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=kernel_size)

        # Calculate gradient magnitude (edge strength)
        # sqrt(dx² + dy²)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        if self.debug:
            self._display_gradients(gradient_x, gradient_y, gradient_magnitude)

        return gradient_x, gradient_y, gradient_magnitude

    def _display_gradients(
        self, grad_x: np.ndarray, grad_y: np.ndarray, grad_mag: np.ndarray
    ):
        """Display Sobel gradient results for debugging."""
        # Normalize for display (0-255 range)
        grad_x_norm = cv2.normalize(
            np.abs(grad_x), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
        grad_y_norm = cv2.normalize(
            np.abs(grad_y), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
        grad_mag_norm = cv2.normalize(
            grad_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(grad_x_norm, cmap="gray")
        plt.title("Sobel X (Vertical Edges)")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(grad_y_norm, cmap="gray")
        plt.title("Sobel Y (Horizontal Edges)")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(grad_mag_norm, cmap="gray")
        plt.title("Gradient Magnitude")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def detect_strong_edges(
        self, gradient_magnitude: np.ndarray, percentile: float = 95
    ) -> np.ndarray:
        """
        Find strong edges using percentile thresholding.

        Args:
            gradient_magnitude: Sobel gradient magnitude
            percentile: Keep top (100-percentile)% strongest edges
                       Higher = more selective (fewer edges)

        Returns:
            Binary mask of strong edges
        """
        # Calculate threshold based on percentile
        # Keeps only the strongest edges (e.g., top 5% if percentile=95)
        threshold_value = np.percentile(gradient_magnitude, percentile)

        # Create binary mask: 1 where edge strength > threshold, 0 otherwise
        edge_mask = (gradient_magnitude > threshold_value).astype(np.uint8) * 255

        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, kernel)

        if self.debug:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.hist(gradient_magnitude.flatten(), bins=50)
            plt.axvline(
                threshold_value,
                color="r",
                linestyle="--",
                label=f"Threshold ({percentile}%)",
            )
            plt.title("Gradient Magnitude Distribution")
            plt.xlabel("Gradient Strength")
            plt.ylabel("Frequency")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.imshow(edge_mask, cmap="gray")
            plt.title(f"Strong Edges (>{threshold_value:.1f})")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        return edge_mask

    def find_border_lines_hough(self, edge_mask: np.ndarray) -> Dict[str, List]:
        """
        Find border lines using Hough Line Transform.

        Args:
            edge_mask: Binary edge mask from Sobel

        Returns:
            Dictionary with detected horizontal and vertical lines
        """
        # Parameters for HoughLinesP:
        # rho: Distance resolution in pixels
        # theta: Angle resolution in radians (π/180 = 1 degree)
        # threshold: Minimum votes (intersections) to detect a line
        # minLineLength: Minimum line length
        # maxLineGap: Maximum gap between line segments to connect them

        lines = cv2.HoughLinesP(
            edge_mask,
            rho=1,
            theta=np.pi / 180,
            threshold=50,  # Adjust based on image size
            minLineLength=min(self.width, self.height)
            // 4,  # At least 1/4 of image dimension
            maxLineGap=20,  # Allow small gaps in border lines
        )

        horizontal_lines = []
        vertical_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate line angle in degrees
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

                # Classify as horizontal (near 0° or 180°) or vertical (near 90°)
                if angle < 30 or angle > 150:  # Horizontal-ish
                    # Average y-position of the line
                    y_avg = (y1 + y2) // 2
                    horizontal_lines.append(
                        {
                            "y": y_avg,
                            "points": [(x1, y1), (x2, y2)],
                            "angle": angle,
                            "length": np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2),
                        }
                    )
                elif 60 < angle < 120:  # Vertical-ish
                    # Average x-position of the line
                    x_avg = (x1 + x2) // 2
                    vertical_lines.append(
                        {
                            "x": x_avg,
                            "points": [(x1, y1), (x2, y2)],
                            "angle": angle,
                            "length": np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2),
                        }
                    )

        # Sort lines by position
        horizontal_lines.sort(key=lambda l: l["y"])
        vertical_lines.sort(key=lambda l: l["x"])

        return {"horizontal": horizontal_lines, "vertical": vertical_lines}

    def select_best_border_lines(
        self, lines_dict: Dict[str, List], border_margin: int = 50
    ) -> Dict[str, Optional[Dict]]:
        """
        Select the best candidate lines for each border side.

        Args:
            lines_dict: Dictionary with horizontal and vertical lines
            border_margin: How close to image edge to look for borders

        Returns:
            Dictionary with best border lines for each side
        """
        horizontals = lines_dict["horizontal"]
        verticals = lines_dict["vertical"]

        best_lines = {"top": None, "bottom": None, "left": None, "right": None}

        # Find top border (horizontal line near top)
        for line in horizontals:
            if line["y"] < border_margin:  # Near top edge
                if (
                    best_lines["top"] is None
                    or line["length"] > best_lines["top"]["length"]
                ):
                    best_lines["top"] = line

        # Find bottom border (horizontal line near bottom)
        for line in horizontals:
            if line["y"] > self.height - border_margin:  # Near bottom edge
                if (
                    best_lines["bottom"] is None
                    or line["length"] > best_lines["bottom"]["length"]
                ):
                    best_lines["bottom"] = line

        # Find left border (vertical line near left)
        for line in verticals:
            if line["x"] < border_margin:  # Near left edge
                if (
                    best_lines["left"] is None
                    or line["length"] > best_lines["left"]["length"]
                ):
                    best_lines["left"] = line

        # Find right border (vertical line near right)
        for line in verticals:
            if line["x"] > self.width - border_margin:  # Near right edge
                if (
                    best_lines["right"] is None
                    or line["length"] > best_lines["right"]["length"]
                ):
                    best_lines["right"] = line

        return best_lines

    def calculate_border_widths(
        self, best_lines: Dict[str, Optional[Dict]]
    ) -> Dict[str, int]:
        """
        Calculate border widths from detected lines.

        Args:
            best_lines: Dictionary with best border lines

        Returns:
            Dictionary with border widths for each side
        """
        border_widths = {"top": 0, "bottom": 0, "left": 0, "right": 0}

        if best_lines["top"]:
            border_widths["top"] = best_lines["top"]["y"]

        if best_lines["bottom"]:
            border_widths["bottom"] = self.height - best_lines["bottom"]["y"]

        if best_lines["left"]:
            border_widths["left"] = best_lines["left"]["x"]

        if best_lines["right"]:
            border_widths["right"] = self.width - best_lines["right"]["x"]

        # If no line detected, try color-based scanning as fallback
        if border_widths["top"] == 0:
            border_widths["top"] = self._scan_color_transition("top")
        if border_widths["bottom"] == 0:
            border_widths["bottom"] = self._scan_color_transition("bottom")
        if border_widths["left"] == 0:
            border_widths["left"] = self._scan_color_transition("left")
        if border_widths["right"] == 0:
            border_widths["right"] = self._scan_color_transition("right")

        return border_widths

    def _scan_color_transition(self, side: str, max_scan: int = 200) -> int:
        """
        Fallback method: Scan for color transition.

        Args:
            side: Which side to scan ('top', 'bottom', 'left', 'right')
            max_scan: Maximum pixels to scan

        Returns:
            Border width in pixels
        """
        if side == "top":
            scan_region = self.gray[0:max_scan, :]
            base_color = np.mean(self.gray[0, :])
        elif side == "bottom":
            scan_region = self.gray[-max_scan:, :]
            base_color = np.mean(self.gray[-1, :])
        elif side == "left":
            scan_region = self.gray[:, 0:max_scan]
            base_color = np.mean(self.gray[:, 0])
        elif side == "right":
            scan_region = self.gray[:, -max_scan:]
            base_color = np.mean(self.gray[:, -1])
        else:
            return 0

        threshold = 30  # Color difference threshold

        for i in range(
            1,
            min(
                max_scan,
                (
                    scan_region.shape[0]
                    if side in ["top", "bottom"]
                    else scan_region.shape[1]
                ),
            ),
        ):
            if side in ["top", "bottom"]:
                current_row = (
                    scan_region[i, :] if side == "top" else scan_region[-i - 1, :]
                )
                diff = np.mean(np.abs(current_row.astype(float) - base_color))
            else:
                current_col = (
                    scan_region[:, i] if side == "left" else scan_region[:, -i - 1]
                )
                diff = np.mean(np.abs(current_col.astype(float) - base_color))

            if diff > threshold:
                return i

        return 0

    def detect_scenario1_borders(self, image_path: str) -> Dict:
        """
        Main detection pipeline for Scenario 1.

        Args:
            image_path: Path to input image

        Returns:
            Dictionary with detection results
        """
        print("\n" + "=" * 50)
        print("SCENARIO 1 DETECTION PIPELINE")
        print("=" * 50)

        # Step 1: Load image
        self.load_image(image_path)

        # Step 2: Apply Sobel gradient detection
        print("✓ Applying Sobel edge detection...")
        grad_x, grad_y, grad_mag = self.apply_sobel_gradient(kernel_size=3)

        # Step 3: Find strong edges
        print("✓ Identifying strong edges...")
        edge_mask = self.detect_strong_edges(grad_mag, percentile=95)

        # Step 4: Find border lines using Hough Transform
        print("✓ Detecting border lines with Hough Transform...")
        lines_dict = self.find_border_lines_hough(edge_mask)

        # Step 5: Select best border lines
        print("✓ Selecting best border candidates...")
        best_lines = self.select_best_border_lines(lines_dict, border_margin=100)

        # Step 6: Calculate border widths
        print("✓ Calculating border widths...")
        border_widths = self.calculate_border_widths(best_lines)

        # Step 7: Calculate image area and corners
        image_area, corners = self._calculate_image_area(border_widths)

        # Step 8: Compile results
        self.results = {
            "border_widths": border_widths,
            "image_area": image_area,
            "corners": corners,
            "best_lines": best_lines,
            "detection_method": "Sobel+Hough",
        }

        self._print_results()

        return self.results

    def _calculate_image_area(
        self, border_widths: Dict[str, int]
    ) -> Tuple[Tuple, List]:
        """
        Calculate image area inside borders.

        Args:
            border_widths: Dictionary with border widths

        Returns:
            Tuple of (image_area, corners)
        """
        x1 = border_widths["left"]
        y1 = border_widths["top"]
        x2 = self.width - border_widths["right"]
        y2 = self.height - border_widths["bottom"]

        # Ensure valid coordinates
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.width, x2)
        y2 = min(self.height, y2)

        image_area = (x1, y1, x2, y2)

        # Calculate corners
        corners = [
            (x1, y1),  # Top-left
            (x2, y1),  # Top-right
            (x2, y2),  # Bottom-right
            (x1, y2),  # Bottom-left
        ]

        return image_area, corners

    def _print_results(self):
        """Print detection results."""
        print("\n" + "=" * 50)
        print("DETECTION RESULTS")
        print("=" * 50)

        bw = self.results["border_widths"]
        print("Border Widths:")
        print(f"  Top:    {bw['top']:4d} px")
        print(f"  Bottom: {bw['bottom']:4d} px")
        print(f"  Left:   {bw['left']:4d} px")
        print(f"  Right:  {bw['right']:4d} px")

        x1, y1, x2, y2 = self.results["image_area"]
        print("\nImage Area:")
        print(f"  Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  Dimensions:  {x2-x1} x {y2-y1} px")
        print(f"  Method:      {self.results['detection_method']}")

    def visualize_detection(self, save_path: str = None):
        """
        Create visualization of detection results.

        Args:
            save_path: Optional path to save visualization image
        """
        if self.image is None:
            print("No image loaded!")
            return

        vis = self.image.copy()

        # Draw border lines
        colors = {
            "top": (0, 255, 0),  # Green
            "bottom": (0, 255, 0),  # Green
            "left": (0, 255, 255),  # Yellow
            "right": (0, 255, 255),  # Yellow
        }

        for side, line_info in self.results["best_lines"].items():
            if line_info:
                color = colors.get(side, (255, 255, 255))
                for (x1, y1), (x2, y2) in [line_info["points"]]:
                    cv2.line(vis, (x1, y1), (x2, y2), color, 3)

        # Draw image area rectangle
        x1, y1, x2, y2 = self.results["image_area"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red

        # Draw corners
        for i, (x, y) in enumerate(self.results["corners"]):
            cv2.circle(vis, (int(x), int(y)), 8, (255, 0, 0), -1)  # Blue
            cv2.putText(
                vis,
                str(i + 1),
                (int(x) + 10, int(y) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

        # Add text overlay
        cv2.putText(
            vis,
            "Scenario 1: Solid Borders",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        bw = self.results["border_widths"]
        cv2.putText(
            vis,
            f"Borders: T={bw['top']}, B={bw['bottom']}, L={bw['left']}, R={bw['right']}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if save_path:
            cv2.imwrite(save_path, vis)
            print(f"\n✓ Visualization saved to: {save_path}")

        return vis

    def extract_subject_area(self) -> np.ndarray:
        """
        Extract the image area (inside borders).

        Returns:
            Cropped image containing just the subject area
        """
        if self.image is None:
            raise ValueError("No image loaded!")

        x1, y1, x2, y2 = self.results["image_area"]

        # Ensure valid coordinates
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.width, x2)
        y2 = min(self.height, y2)

        if x2 <= x1 or y2 <= y1:
            print("Warning: Invalid image area, returning original image")
            return self.image.copy()

        return self.image[y1:y2, x1:x2].copy()


# ============================================================================
# USAGE EXAMPLES AND TESTING
# ============================================================================


def test_scenario1_single_image():
    """Test the detector on a single image."""

    # Create detector with debug mode ON
    detector = Scenario1Detector(debug=True)

    # Test image path (update this to your image)
    test_image = r"C:\Users\S10DIGITAL\python\py_GIMP\openCV\trial_Directory\Pic4.jpeg"

    try:
        # Run detection
        results = detector.detect_scenario1_borders(test_image)

        # Visualize results
        vis = detector.visualize_detection()

        # Extract subject area
        subject = detector.extract_subject_area()

        # Display results
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(detector.image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Detection Results")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(subject, cv2.COLOR_BGR2RGB))
        plt.title("Extracted Subject")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        return results

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


def analyze_image_properties(image_path: str):
    """Analyze image properties to understand why detection might fail."""

    image = cv2.imread(image_path)
    if image is None:
        print("Cannot load image")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    print("\n" + "=" * 50)
    print("IMAGE ANALYSIS")
    print("=" * 50)

    # Analyze border regions
    border_size = 50
    top_region = gray[0:border_size, :]
    bottom_region = gray[h - border_size : h, :]
    left_region = gray[:, 0:border_size]
    right_region = gray[:, w - border_size : w]

    regions = {
        "Top": top_region,
        "Bottom": bottom_region,
        "Left": left_region,
        "Right": right_region,
    }

    print("\nBorder Region Analysis:")
    print("-" * 40)
    for name, region in regions.items():
        mean_val = np.mean(region)
        std_val = np.std(region)
        uniformity = std_val / mean_val if mean_val > 0 else 0

        print(
            f"{name:8} | Mean: {mean_val:6.1f} | Std: {std_val:6.1f} | Uniformity: {uniformity:.3f}"
        )

        # Check if border is uniform (low std)
        if std_val < 20:
            print("  -> Likely solid border (uniform)")
        else:
            print("  -> Not uniform, might not be a solid border")

    # Check overall contrast
    overall_std = np.std(gray)
    print(f"\nOverall image std: {overall_std:.1f}")
    if overall_std < 30:
        print("Warning: Low contrast image - edge detection may be difficult")

    # Show histogram
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(gray.flatten(), bins=50, alpha=0.7)
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    How to use this script:

    1. For testing a single image:
       - Update the test_image path below
       - Run: python scenario1_detector.py

    2. For analyzing image properties:
       - Call analyze_image_properties() with your image path
    """

    # Example 1: Test detection
    print("Testing Scenario 1 detector...")
    results = test_scenario1_single_image()

    # Example 2: Analyze image (if detection fails)
    # image_path = r"C:\path\to\your\image.jpg"
    # analyze_image_properties(image_path)
