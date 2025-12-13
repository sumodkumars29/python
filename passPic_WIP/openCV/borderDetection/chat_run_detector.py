import sys
import cv2
from chat_border_detector_1 import BorderDetector


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_detector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    detector = BorderDetector(image_path, debug=True)

    detector.load()
    fb = detector.detect_face()
    print("Face box:", fb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # result = detector.process()

    # warped = result["warped"]
    # corners = result["corners"]
    #
    # print("Corners:", corners)
    #
    # if warped is not None:
    #     cv2.imshow("Warped", warped)
    #     cv2.waitKey(0)


if __name__ == "__main__":
    main()
