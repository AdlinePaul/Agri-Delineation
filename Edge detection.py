import cv2
import numpy as np
from google.colab.patches import cv2_imshow

image_path = "/content/t4.jpg"
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not load image")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.bilateralFilter(gray, 9, 75, 75)
edged = cv2.Canny(blurred, 100, 250)
edged = cv2.dilate(edged, None, iterations=1)

contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    prominent_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(image, [prominent_contour], -1, (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(prominent_contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # --- Reference Object Based Calculation ---
    reference_object_height_pixels =  211# Replace with pixel height of reference object
    reference_object_height_cm =  365.76# Replace with actual height of reference object in cm

    pixel_to_cm_ratio = reference_object_height_cm / reference_object_height_pixels
    object_height_cm = h * pixel_to_cm_ratio

    # print(object_height_cm)
    width = w-x
    px = 0.0264583333
    width_cm = round(width*px,2)
    cv2.putText(image, f"Width: {width_cm} cm", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(image, f"Height: {round(object_height_cm,2)} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2_imshow(image)
else:
    print("No contours found in the image.")
