import numpy as np
import cv2
from matplotlib import pyplot as plt


def extract_background(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Create a mask of zeros with the same dimensions as the image
    mask = np.zeros(img.shape[:2],np.uint8)

    # Specify background and foreground models
    bgd_model = np.zeros((1,65),np.float64)
    fgd_model = np.zeros((1,65),np.float64)

    # Define the rectangle to start with GrabCut algorithm
    rect = (50,50,img.shape[1]-50,img.shape[0]-50)  # x, y, width, height

    # Apply GrabCut algorithm
    cv2.grabCut(img,mask,rect,bgd_model,fgd_model,5,cv2.GC_INIT_WITH_RECT)

    # Create a mask where all probable background pixels are marked as 0 and others as 1
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    # Multiply the original image with the mask to get the background
    background = img * mask2[:, :, np.newaxis]

    # Show the images
    plt.subplot(1,2,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,2,2),plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    plt.title('Extracted Background'), plt.xticks([]), plt.yticks([])
    plt.show()


# Example usage
image_path = ('/home/jafar/sceneGen/Scene-Generator/results/test_3/results_2/image.png'
              ''
              '`')
extract_background(image_path)
