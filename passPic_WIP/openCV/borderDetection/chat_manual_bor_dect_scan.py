from pathlib import Path
import cv2

import pandas as pd


def scan_top_left_quadrant(image_path, variance_threshold=200, debug=False):
    # Scan top-left quadrant of the image to detect edges based on colour variance

    # Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    h, w = image.shape[:2]

    # Define scan region parameters
    scan_width_ratio = 0.30
    scan_height_ratio = 0.25

    # Calculate region bounds
    scan_w = int(w * scan_width_ratio)
    scan_h = int(h * scan_height_ratio)
    start_x = 0
    start_y = 1130
    end_y = scan_h

    # Extract region of interest
    # roi = image[start_y:end_y, start_x:scan_w]
    # roi = image[0:150, 0:150]
    roi = image[1130:h, 0:150]

    # Store variance points
    x_variance = []
    y_variance = []
    events = []

    # Horizontal Scan (row by row)
    for y in range(roi.shape[0]):
        # Initialize with first pixel in this row
        prev_pixel = roi[y, 0]
        b1, g1, r1 = prev_pixel

        for x in range(1, roi.shape[1]):
            curr_pixel = roi[y, x]
            b2, g2, r2 = curr_pixel

            # Calculate total colour difference
            diff = (
                abs(int(b1) - int(b2)) + abs(int(g1) - int(g2)) + abs(int(r1) - int(r2))
            )

            # Absolute co-ordinates in original image
            abs_x = x + start_x
            abs_y = y + start_y

            if diff > variance_threshold:
                # Edge detected
                x_variance.append(abs_x)
                events.append(
                    {
                        "x": abs_x,
                        "y": abs_y,
                        "b": int(b2),
                        "g": int(g2),
                        "r": int(r2),
                        "diff": diff,
                        "scan": "H",
                        "edge": True,
                    }
                )
                prev_pixel = curr_pixel  # Reset comparison baseline
                b1, g1, r1 = b2, g2, r2
            # else:
            #     # No edge - still log for debugging
            #     events.append(
            #         {
            #             "x": abs_x,
            #             "y": abs_y,
            #             "b": int(b2),
            #             "g": int(g2),
            #             "r": int(r2),
            #             "diff": diff,
            #             "scan": "H",
            #             "edge": False,
            #         }
            #     )

    # Vertical Scan (column by column)
    for x in range(roi.shape[1]):
        # Initialize with first pixel in this column
        prev_pixel = roi[0, x]
        b1, g1, r1 = prev_pixel

        for y in range(1, roi.shape[0]):
            curr_pixel = roi[y, x]
            b2, g2, r2 = curr_pixel

            # Calculate total colour difference
            diff = (
                abs(int(b1) - int(b2)) + abs(int(g1) - int(g2)) + abs(int(r1) - int(r2))
            )

            # Absolute co-ordinates in original image
            abs_x = x + start_x
            abs_y = y + start_y

            if diff > variance_threshold:
                # Edge detected
                y_variance.append(abs_y)
                events.append(
                    {
                        "x": abs_x,
                        "y": abs_y,
                        "b": int(b2),
                        "g": int(g2),
                        "r": int(r2),
                        "diff": diff,
                        "scan": "V",
                        "edge": True,
                    }
                )
                prev_pixel = curr_pixel  # Reset comparison baseline
                b1, g1, r1 = b2, g2, r2
            # else:
            #     # No edge - still log for debugging
            #     events.append(
            #         {
            #             "x": abs_x,
            #             "y": abs_y,
            #             "b": int(b2),
            #             "g": int(g2),
            #             "r": int(r2),
            #             "diff": diff,
            #             "scan": "V",
            #             "edge": False,
            #         }
            #     )

    # Create DataFrame
    df = pd.DataFrame(events)

    # Save results
    output_path = Path(image_path).parent / "top_left_scan_variance.xlsx"
    df.to_excel(output_path, index=False, engine="openpyxl")

    # Calculate intersection points
    if x_variance and y_variance:
        quad_point = (min(x_variance), max(y_variance))
        if debug:
            print(f"ROI: {start_x}:{scan_w}, {start_y};{end_y}")
            print(f"X variances found: {len(x_variance)}")
            print(f"Y variances found: {len(y_variance)}")
            print(f"Estimated Quad_point: {quad_point}")
            print(f"Data saved to: {output_path}")
        return quad_point
    else:
        print("No edges detected above threshold")
        return None


# Example usage
if __name__ == "__main__":
    try:
        result = scan_top_left_quadrant(
            r"C:\Users\S10DIGITAL\python\py_GIMP\openCV\trial_Directory\Pic10-ValueInvert.jpeg",
            variance_threshold=200,
            debug=True,
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
