import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Define the labels for COCO dataset
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def detect_objects(image_path, threshold=0.5):
    # Load image
    img = Image.open(image_path)
    img_tensor = F.to_tensor(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        prediction = model(img_tensor)

    # Filter out detections below threshold
    boxes = prediction[0]['boxes'][prediction[0]['scores'] > threshold]
    labels = prediction[0]['labels'][prediction[0]['scores'] > threshold]
    masks = prediction[0]['masks'][prediction[0]['scores'] > threshold]

    # Visualize
    img = Image.open(image_path)
    img_np = np.array(img)
    plt.imshow(img_np)
    ax = plt.gca()
    for box, label, mask in zip(boxes, labels, masks):
        x1, y1, x2, y2 = box
        category = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
        plt.text(x1, y1, f'{category}', bbox=dict(facecolor='red', alpha=0.5), fontsize=12, color='white')
        img_np[mask == 255] = (img_np[mask == 255] * 0.5) + (np.array([255, 0, 0]) * 0.5)
    plt.imshow(img_np)
    plt.axis('off')
    plt.show()


# Example usage
image_path = '/home/jafar/sceneGen/Scene-Generator/living room.png'
detect_objects(image_path)
