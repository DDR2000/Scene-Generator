import torch
import os
import cv2
import json
from torchvision.models.detection import retinanet
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Function to extract depth within bounding boxes
def extract_depth_within_boxes(depth_map, objects_info):
    depths = []
    for obj_idx, obj in objects_info.items():
        print(f"processing object {obj_idx}")
        xmin, ymin, xmax, ymax = map(int, obj["box"])  # Convert to integers
        box_depth = depth_map[ymin:ymax, xmin:xmax].mean()
        depths.append(box_depth)
    return depths


# Load RetinaNet model
# Load the pre-trained RetinaNet model
retina_net_model = retinanet.retinanet_resnet50_fpn(pretrained=True)

# Set the model to evaluation mode
retina_net_model.eval()

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


def run_retina_net(image_path, save_path, threshold=0.5):
    # Load image
    img = Image.open(image_path)
    # Resize image to 1024x1024
    img = img.resize((1024, 1024))

    img_tensor = F.to_tensor(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        prediction = retina_net_model(img_tensor)

    # Filter out detections below threshold
    boxes = prediction[0]['boxes'][prediction[0]['scores'] > threshold]
    labels = prediction[0]['labels'][prediction[0]['scores'] > threshold]
    scores = prediction[0]['scores'][prediction[0]['scores'] > threshold]

    # Convert PyTorch tensors to NumPy arrays
    boxes_c = boxes.cpu().numpy()
    labels_c = labels.cpu().numpy()
    scores_c = scores.cpu().numpy()

    # Organize detection information in a dictionary
    return_retina_net_objects_info = {}

    for j, box in enumerate(boxes_c):
        obj_info = {
            "class": COCO_INSTANCE_CATEGORY_NAMES[labels_c[j]],
            "box": box.tolist()
        }
        return_retina_net_objects_info[str(j)] = obj_info

    # Visualize
    plt.figure(figsize=(10.24, 10.24))  # Set figure size directly

    plt.imshow(img)

    retina_net_objects_info = list(zip(boxes, labels, scores))
    for box, label, score in retina_net_objects_info:
        x1, y1, x2, y2 = box
        # Scale bounding box coordinates to match the resized image
        x1_scaled = x1 * 1024 / img.width
        x2_scaled = x2 * 1024 / img.width
        y1_scaled = y1 * 1024 / img.height
        y2_scaled = y2 * 1024 / img.height

        plt.gca().add_patch(plt.Rectangle((x1_scaled, y1_scaled), x2_scaled - x1_scaled, y2_scaled - y1_scaled,
                                   fill=False, edgecolor='red', linewidth=2))
        category = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
        plt.text(x1_scaled, y1_scaled, f'{category} {score:.2f}', bbox=dict(facecolor='red', alpha=0.5),
                 fontsize=12, color='white')

    plt.axis('off')

    # Save figure with appropriate size
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)

    plt.close()  # Close the figure to release resources

    return return_retina_net_objects_info



# add {i}.png
filenames = [
    "presentation/sdxl/sdxl_image",
    "presentation/photo_sdxl/sdxl_photo_image",
    "presentation/sdxl/reimagine/reimagine_image",
    "presentation/photo_sdxl/reimagine/photo_reimagine_image",
    "presentation/sdxl/canny_conditioning/canny_conditioning_image",
    "presentation/photo_sdxl/canny_conditioning/canny_conditioning_image",
    "presentation/semantic_sd15/upscaled/upscaled_semantic_sd15_image",
    "presentation/semantic_sd15/control_net/cnet_generated_image"
           ]

for file in filenames:
    # Find the index of the last occurrence of '/'
    last_slash_index = file.rfind('/')

    # If '/' is found, remove everything after it (including '/')
    if last_slash_index != -1:
        directory_path = file[:last_slash_index]
        retina_path = f"{directory_path}/retina"
        create_directory(retina_path)
    else:
        # Dump in temp
        print("wrong paths provided")
        create_directory("presentation/temp")
        retina_path = "presentation/temp"
        directory_path = retina_path

    for i in range(9):
        if "cnet_generated_image" in file:
            for cond in ["Boy", "Girl", "Man", "Women"]:
                filename = f"{file}_{i}_{cond}.png"
                save_path = f"{retina_path}/retina_image_{i}_{cond}.png"
                print("processing: ", filename)
                # Run RetinaNet
                objects_info = run_retina_net(filename, save_path, threshold=0.5)
                print("retina_objects_info")
                print(objects_info)
                # Load depth map image into numpy array.
                depth_map_image_path = f"{directory_path}/depth_map/depth_image_{i}_{cond}.png"

                # Open the image
                image = Image.open(depth_map_image_path)

                # Upscale the image to 1024x1024
                image = image.resize((1024, 1024))

                # Save the upscaled image
                image.save(depth_map_image_path)

                # Load the depth map image using OpenCV
                depth_map_image = cv2.imread(depth_map_image_path, cv2.IMREAD_UNCHANGED)

                # Convert the depth map image to a NumPy array
                depth_map_array = np.array(depth_map_image)

                # Calculate minimum and maximum depth values on the depth map
                min_depth = depth_map_array.min()
                max_depth = depth_map_array.max()

                # Calculate the mean depth for each bounding box
                depths = extract_depth_within_boxes(depth_map_array, objects_info)

                # Create a list to store the coordinates of detected objects in 3D space
                object_coordinates_3d = []

                # Define the size of the virtual 3D space (adjust as needed)
                virtual_space_size = (1024, 1024, 1024)

                # Loop through detected objects and convert their coordinates
                for obj_idx, obj in objects_info.items():
                    xmin, ymin, xmax, ymax = map(int, obj["box"])
                    box_depth = depths[int(obj_idx)]

                    # Calculate the x and y coordinates based on the bounding box
                    x = (xmin + xmax) / 2
                    y = (ymin + ymax) / 2

                    # Map the depth value to the 3D space (adjust scale factor as needed)
                    z = (box_depth - min_depth) / (max_depth - min_depth) * virtual_space_size[2]

                    # Append the object's 3D coordinates to the list
                    object_coordinates_3d.append({
                        "name": objects_info[obj_idx]["class"],
                        "x": x,
                        "y": y,
                        "z": z
                    })

                print("object_coordinates_3d")
                print(object_coordinates_3d)

                # Save the object coordinates in 3D space as a JSON file
                json_file_path = f"{retina_path}/object_coordinates_3d_{i}_{cond}.json"
                with open(json_file_path, "w") as json_file:
                    json.dump(object_coordinates_3d, json_file, indent=4)

                print(f"Processed results for set {i}_{cond} in {directory_path}")
        else:
            filename = f"{file}_{i}.png"
            save_path = f"{retina_path}/retina_image_{i}.png"
            print("processing: ", filename)
            # Run RetinaNet
            objects_info = run_retina_net(filename, save_path, threshold=0.5)
            print("retina_objects_info")
            print(objects_info)
            # Load depth map image into numpy array.
            depth_map_image_path = f"{directory_path}/depth_map/depth_image_{i}.png"

            if ("reimagine" in filename) | ("upscaled" in filename):
                # Open the image
                image = Image.open(depth_map_image_path)

                # Upscale the image to 1024x1024
                image = image.resize((1024, 1024))

                # Save the upscaled image
                image.save(depth_map_image_path)

            # Load the depth map image using OpenCV
            depth_map_image = cv2.imread(depth_map_image_path, cv2.IMREAD_UNCHANGED)

            # Convert the depth map image to a NumPy array
            depth_map_array = np.array(depth_map_image)

            # Calculate minimum and maximum depth values on the depth map
            min_depth = depth_map_array.min()
            max_depth = depth_map_array.max()

            # Calculate the mean depth for each bounding box
            depths = extract_depth_within_boxes(depth_map_array, objects_info)

            # Create a list to store the coordinates of detected objects in 3D space
            object_coordinates_3d = []

            # Define the size of the virtual 3D space (adjust as needed)
            virtual_space_size = (1024, 1024, 1024)

            # Loop through detected objects and convert their coordinates
            for obj_idx, obj in objects_info.items():
                xmin, ymin, xmax, ymax = map(int, obj["box"])
                box_depth = depths[int(obj_idx)]

                # Calculate the x and y coordinates based on the bounding box
                x = (xmin + xmax) / 2
                y = (ymin + ymax) / 2

                # Map the depth value to the 3D space (adjust scale factor as needed)
                z = (box_depth - min_depth) / (max_depth - min_depth) * virtual_space_size[2]

                # Append the object's 3D coordinates to the list
                object_coordinates_3d.append({
                    "name": objects_info[obj_idx]["class"],
                    "x": x,
                    "y": y,
                    "z": z
                })

            print("object_coordinates_3d")
            print(object_coordinates_3d)

            # Save the object coordinates in 3D space as a JSON file
            json_file_path = f"{retina_path}/object_coordinates_3d_{i}.json"
            with open(json_file_path, "w") as json_file:
                json.dump(object_coordinates_3d, json_file, indent=4)

            print(f"Processed results for set {i} in {directory_path}")

    print("Processing completed.")
