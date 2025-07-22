import cv2
import numpy as np
from hand_detector.detector import YOLO
from unified_detector import Fingertips

# Initialize the YOLO hand detector with pre-trained weights and a detection threshold.
hand = YOLO(weights='weights/yolo.h5', threshold=0.8)

# Initialize the Fingertips model for gesture classification and fingertip regression.
fingertips = Fingertips(weights='weights/fingertip.h5')

# Read the input image.
image = cv2.imread('data/sample.jpg')

# Detect the hand in the image.
# tl: top-left corner (x, y) of the detected hand bounding box
# br: bottom-right corner (x, y) of the detected hand bounding box
tl, br = hand.detect(image=image)

# If a hand is detected (i.e., tl or br is not None)
if tl or br is not None:
    # Crop the image to the detected hand region.
    # image in opencv use y,x order
    cropped_image = image[tl[1]:br[1], tl[0]: br[0]]
    height, width, _ = cropped_image.shape

    # Use the Fingertips model to classify the gesture and regress fingertip positions.
    # prob: probabilities for each fingertip being present (e.g., [0.9, 0.1, ...])
    # pos: predicted normalized (x, y) positions for each fingertip (values in [0, 1])
    # Example output prob and pos:
    # prob: [0.9, 0.1, 0.1, 0.1, 0.1]
    # pos: [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1]]
    prob, pos = fingertips.classify(image=cropped_image)
    pos = np.mean(pos, 0)  # Average position of each finger (column wise)

    # Post-processing:
    # Convert probabilities to binary (1 if >= 0.5, else 0)
    prob = np.asarray([(p >= 0.5) * 1.0 for p in prob])
    # Convert normalized positions to absolute image coordinates
    for i in range(0, len(pos), 2):
        pos[i] = pos[i] * width + tl[0]      # x-coordinate
        pos[i + 1] = pos[i + 1] * height + tl[1]  # y-coordinate

    # Drawing:
    index = 0
    color = [(15, 15, 240), (15, 240, 155), (240, 155, 15), (240, 15, 155), (240, 15, 240)]
    # Draw the hand bounding box
    image = cv2.rectangle(image, (tl[0], tl[1]), (br[0], br[1]), (235, 26, 158), 2)
    # Draw circles for each detected fingertip
    for c, p in enumerate(prob):
        if p > 0.5:
            image = cv2.circle(image, (int(pos[index]), int(pos[index + 1])), radius=12, color=color[c], thickness=-2)
        index = index + 2

    # Display the result
    cv2.imshow('Unified Gesture & Fingertips Detection', image)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
