#!/usr/bin/env python3
"""
Trigger script for passport border detection.
Usage: python ds_border_detect_trigger.py image_path output_path debug
Example: python ds_border_detect_trigger.py "C:\path\to\image.jpg" "C:\output\folder" True
"""

import sys
from pathlib import Path
import cv2


def main(image_path: str, output_path: str, debug: bool = False):
    """
    Main processing function.

    Args:
        image_path: Path to input image
        output_path: Directory to save results
        debug: Enable debug output
    """

    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))

    try:
        from ds_bor_dect import PassportBorderDetector
    except ImportError:
        print("ERROR: Could not import detect_borders.py")
        print("Make sure detect_borders.py is in the same directory as this script")
        sys.exit(1)

    print(f"Processing: {image_path}")
    print(f"Output directory: {output_path}")
    print(f"Debug mode: {debug}")
    print("-" * 50)

    # Create output directory if it doesn't exist
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create detector and process image
        detector = PassportBorderDetector(debug=debug)
        results = detector.detect_all_borders(image_path)

        # Print key results
        print("\n" + "=" * 50)
        print("DETECTION RESULTS:")
        print("=" * 50)
        print(f"Scenario: {results['scenario']}")
        print(f"Confidence: {results['confidence']:.2%}")
        print(
            f"Original size: {results['original_size'][0]} x {results['original_size'][1]}"
        )
        print(f"Skew detected: {results['is_skewed']}")

        if results["is_skewed"]:
            print(f"Skew angle: {results['skew_angle']:.2f}°")

        if results["outer_border"]:
            print("\nBorder widths:")
            borders = results["outer_border"]
            for side, width in borders.items():
                print(f"  {side}: {width}px")

        if results["image_area"]:
            x1, y1, x2, y2 = results["image_area"]
            print(f"\nImage area: ({x1}, {y1}) to ({x2}, {y2})")
            print(f"Image size: {x2-x1} x {y2-y1} pixels")

        # Generate visualization
        vis_image = detector.visualize_results()

        # Save visualization
        vis_path = output_dir / "detection_results.jpg"
        cv2.imwrite(str(vis_path), vis_image)
        print(f"\nVisualization saved to: {vis_path}")

        # Extract and save subject area
        subject = detector.extract_subject_area()
        subject_path = output_dir / "subject_area.jpg"
        cv2.imwrite(str(subject_path), subject)
        print(f"Subject area saved to: {subject_path}")

        # Save rectified image if skewed
        if results["is_skewed"]:
            rectified, mask = detector.rectify_image()
            rectified_path = output_dir / "rectified.jpg"
            mask_path = output_dir / "rectification_mask.jpg"
            cv2.imwrite(str(rectified_path), rectified)
            cv2.imwrite(str(mask_path), mask)
            print(f"Rectified image saved to: {rectified_path}")

        print(f"\nAll results saved to: {output_dir}")

    except Exception as e:
        print(f"\nERROR during processing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 3:
        print(
            "Usage: python ds_border_detect_trigger.py image_path output_path [debug]"
        )
        print(
            "Example: python ds_border_detect_trigger.py image.jpg output_folder True"
        )
        print("\nArguments:")
        print("  image_path  : Path to input image")
        print("  output_path : Directory to save results")
        print("  debug       : True or False (optional, default: False)")
        sys.exit(1)

    # Get arguments
    image_path = sys.argv[1]
    output_path = sys.argv[2]

    # Parse debug flag (optional, default to False)
    debug = False
    if len(sys.argv) >= 4:
        debug_str = sys.argv[3].lower()
        if debug_str in ["true", "t", "1", "yes", "y"]:
            debug = True

    # Run main function
    main(image_path, output_path, debug)
