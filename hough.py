import os
import cv2
import numpy as np


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


create_directory("presentation/sdxl/hough")
create_directory("presentation/photo_sdxl/hough")

create_directory("presentation/sdxl/reimagine/hough")
create_directory("presentation/photo_sdxl/reimagine/hough")

create_directory("presentation/sdxl/canny_conditioning/hough")
create_directory("presentation/photo_sdxl/canny_conditioning/hough")

create_directory("presentation/semantic_sd15/upscaled/hough")
def run_hough_transform(image_path, saving_path):  # Try dotted line, or transparent lines
    # Load the image
    hough_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Edge detection using Canny
    edges = cv2.Canny(hough_image, 50, 150)  # Adjust the thresholds as needed

    # Apply Hough line transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 400)  # Adjust parameters as needed

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
            cv2.line(hough_image, pt1, pt2, (0, 0, 255), 2)

        # Compute the predominant orientation of the object
        orientation = np.median(angles)
        print("Predominant orientation:", orientation)

        # Save image to disk
        cv2.imwrite(saving_path, hough_image)
        print(f"Result image saved to: {saving_path}")

        # Show the result
        # cv2.imshow('Image with Hough Lines', hough_image)

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    else:
        print("No lines detected.")
        orientation = None

    return hough_image, orientation


for i in range(9):
    # Do for SDXL folder
    filename = f"presentation/sdxl/sdxl_image_{i}.png"
    saving_path = f"presentation/sdxl/hough/hough_sdxl_image_{i}.png"
    #Find object orientation, find orientation of extracted objects
    hough_image, predominant_orientation = run_hough_transform(filename, saving_path)
    print("predominant_orientation: ", predominant_orientation)

    # Do for photo sdxl
    filename = f"presentation/photo_sdxl/sdxl_photo_image_{i}.png"
    saving_path = f"presentation/photo_sdxl/hough/hough_sdxl_image_{i}.png"
    # Find object orientation, find orientation of extracted objects
    hough_image, predominant_orientation = run_hough_transform(filename, saving_path)
    print("predominant_orientation: ", predominant_orientation)



    # Do for SDXL.reimagine folder
    filename = f"presentation/sdxl/reimagine/reimagine_image_{i}.png"
    saving_path = f"presentation/sdxl/reimagine/hough/hough_image_{i}.png"
    # Find object orientation, find orientation of extracted objects
    hough_image, predominant_orientation = run_hough_transform(filename, saving_path)
    print("predominant_orientation: ", predominant_orientation)

    # Do for photo_sdxl.reimagine
    filename = f"presentation/photo_sdxl/reimagine/photo_reimagine_image_{i}.png"
    saving_path = f"presentation/photo_sdxl/reimagine/hough/hough_image_{i}.png"
    # Find object orientation, find orientation of extracted objects
    hough_image, predominant_orientation = run_hough_transform(filename, saving_path)
    print("predominant_orientation: ", predominant_orientation)



    # Do for SDXL.canny_conditioning folder
    filename = f"presentation/sdxl/canny_conditioning/canny_conditioning_image_{i}.png"
    saving_path = f"presentation/sdxl/canny_conditioning/hough/hough_image_{i}.png"
    # Find object orientation, find orientation of extracted objects
    hough_image, predominant_orientation = run_hough_transform(filename, saving_path)
    print("predominant_orientation: ", predominant_orientation)

    # Do for SDXL.canny_conditioning
    filename = f"presentation/photo_sdxl/canny_conditioning/canny_conditioning_image_{i}.png"
    saving_path = f"presentation/photo_sdxl/canny_conditioning/hough/hough_sdxl_image_{i}.png"
    # Find object orientation, find orientation of extracted objects
    hough_image, predominant_orientation = run_hough_transform(filename, saving_path)
    print("predominant_orientation: ", predominant_orientation)


    # Do for SDXL folder
    filename = f"presentation/semantic_sd15/upscaled/upscaled_semantic_sd15_image_{i}.png"
    saving_path = f"presentation/semantic_sd15/upscaled/hough/hough_image_{i}.png"
    #Find object orientation, find orientation of extracted objects
    hough_image, predominant_orientation = run_hough_transform(filename, saving_path)
    print("predominant_orientation: ", predominant_orientation)

