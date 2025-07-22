import cv2
import numpy as np
from hand_detector.detector import YOLO
from unified_detector import Fingertips

def crop_fingertip(image):
    """
    Detect the first fingertip in the image and return a 100x100 crop around its center.
    If no fingertip is detected, return None.
    Args:
        image (np.ndarray): Input image (BGR, as loaded by cv2)
    Returns:
        np.ndarray or None: Cropped fingertip image or None if not detected
    """
    if image is None or len(image.shape) != 3:
        return None

    height, width, _ = image.shape

    fingertips = Fingertips(weights='weights/fingertip.h5')
    prob, pos = fingertips.classify(image=image)

    print(prob)
    print(pos)

    if pos is None or len(pos) < 2:
        return None

    pos = np.mean(pos, axis=0)
    prob = (np.asarray(prob) >= 0.5).astype(float)
    print("pos", pos)
    print("prob", prob)

    first_idx = None
    for i, p in enumerate(prob):
        if p > 0.5:
            first_idx = i
            break
    if first_idx is None:
        return None

    x = int(pos[2 * first_idx] * width)
    y = int(pos[2 * first_idx + 1] * height)

    # Optional: draw circle for debug
    for i, p in enumerate(prob):
        if p > 0.5:
            x = pos[2*i] * width
            y = pos[2*i+1] * height
            x, y = int(x), int(y)
            # annotate
            cv2.putText(image, str(i) + " " + str(x) + ", " + str(y), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), 2)
    cv2.imshow('Fingertip', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Crop 100x100 around (x, y)
    half_size = 50
    x1 = max(x - half_size, 0)
    y1 = max(y - half_size, 0)
    x2 = min(x + half_size, width)
    y2 = min(y + half_size, height)

    cropped = image[y1:y2, x1:x2]

    # Pad if smaller than 100x100
    h, w = cropped.shape[:2]
    if h != 100 or w != 100:
        cropped = cv2.copyMakeBorder(
            cropped,
            top=0,
            bottom=100 - h,
            left=0,
            right=100 - w,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

    return cropped
