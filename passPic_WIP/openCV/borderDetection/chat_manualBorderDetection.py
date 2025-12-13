from pathlib import Path
import cv2
import pandas as pd


def scan_top_right_quadrant(image_path):
    # Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        print("Failed to load image")
        return

    h, w = image.shape[:2]

    # Extract the top-right quadrant
    quad = image[0 : h // 2, w // 2 : w]

    data = []

    # Scan every pixel
    for y in range(quad.shape[0]):
        for x in range(quad.shape[1]):
            b, g, r = quad[y, x]
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)  # grayscale value
            data.append([x, y, int(b), int(g), int(r), gray])

    # Create DataFrame
    df = pd.DataFrame(data, columns=["x", "y", "b", "g", "r", "gray"])  # type: ignore

    # Save output files in the same folder as input
    out_xlsx = Path(image_path).parent / "top_right_scan.xlsx"

    df.to_excel(out_xlsx, index=False, engine="openpyxl")

    print(f"Scan data saved to :\n{out_xlsx}")


# Example usage
scan_top_right_quadrant(
    r"C:\Users\S10DIGITAL\python\py_GIMP\openCV\trial_directory\Pic10.jpeg"
)
