import cv2
import numpy as np


# Load the image
image = cv2.imread('/home/jafar/sceneGen/Scene-Generator/test_1/results_0/image.png', cv2.IMREAD_GRAYSCALE)


# Edge detection using Canny
edges = cv2.Canny(image, 50, 150)  # Adjust the thresholds as needed


# Apply Hough line transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)  # Adjust parameters as needed


if lines is not None:
    # Initialize an empty list to store line angles
    angles = []

    for rho, theta in lines[:, 0]:
        # Convert Hough line representation to Cartesian coordinates
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        # Compute line angle
        angle = np.degrees(theta) - 90  # Convert radians to degrees and adjust for reference axis
        angles.append(angle)

        # Plot Hough lines on the original image (optional)
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(image, pt1, pt2, (0, 0, 255), 2)

    # Compute the predominant orientation of the object
    predominant_orientation = np.median(angles)
    print("Predominant orientation:", predominant_orientation)

    # Show the result
    cv2.imshow('Image with Hough Lines', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No lines detected.")
