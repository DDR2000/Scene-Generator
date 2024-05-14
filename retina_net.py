import torch
import torchvision
from torchvision.models.detection import retinanet
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained RetinaNet model
model = retinanet.retinanet_resnet50_fpn(pretrained=True)

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
    scores = prediction[0]['scores'][prediction[0]['scores'] > threshold]

    # Visualize
    img = Image.open(image_path)
    plt.imshow(img)
    ax = plt.gca()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
        category = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
        plt.text(x1, y1, f'{category} {score:.2f}', bbox=dict(facecolor='red', alpha=0.5), fontsize=12, color='white')
    plt.axis('off')
    plt.show()

# Example usage
image_path = '/home/jafar/sceneGen/Scene-Generator/living room.png'
detect_objects(image_path)
