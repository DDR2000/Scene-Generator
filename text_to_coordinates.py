import os
import torch
import cv2
import urllib.request
import matplotlib.image
from diffusers import StableDiffusionXLPipeline
import shutil
import json
import numpy as np


# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Get the directory name for results from the user
results_directory = input("Enter the results directory name: ")
results_directory = results_directory.strip()
create_directory(results_directory)

# Load Stable Diffusion XL
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load MiDaS model
model_type = "DPT_Large"  # Change to your desired MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# Read prompts from a file
prompt_file = "prompts3.txt"  # Replace with the path to your prompt file
with open(prompt_file, "r") as file:
    prompts = file.read().splitlines()


# Function to extract depth within bounding boxes
def extract_depth_within_boxes(depth_map, boxes):
    depths = []
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)  # Convert to integers
        box_depth = depth_map[ymin:ymax, xmin:xmax].mean()
        depths.append(box_depth)
    return depths


# Create a list to store all detected objects
detected_objects = []

# Process each set of prompts
for i, prompt in enumerate(prompts):
    # Create a directory for the current set of results
    set_directory = os.path.join(results_directory, f"results_{i}")
    create_directory(set_directory)

    # Generate an image using Stable Diffusion XL
    image = pipe(prompt + " canon50, focal length 50mm, sensor size 35 mm").images[0]
    image.save(f"{set_directory}/image.png")

    # Detect objects and bounding boxes using YOLO
    imgs = [f"{set_directory}/image.png"]
    results = model(imgs)
    results.save()

    # Load YOLO results into a pandas DataFrame
    yolo_results_df = results.pandas().xyxy[0]

    # Extract object names and store them in a list
    object_names = yolo_results_df["name"].tolist()

    # Move inference output to specified output folder
    timestamp = os.listdir('./runs/detect/')[0]
    shutil.move(os.path.join('./runs/detect', timestamp), os.path.join(set_directory, timestamp))

    # Load the generated image for MiDaS
    filename = f"{set_directory}/image.png"
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = midas_transforms.dpt_transform(img).to(device)

    # Generate a depth map using MiDaS
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    output = prediction.cpu().numpy()

    # Calculate minimum and maximum depth values on the depth map
    min_depth = output.min()
    max_depth = output.max()

    # Save the depth map
    matplotlib.image.imsave(f"{set_directory}/depth.png", output)

    # Extract the bounding boxes from YOLO results
    boxes = yolo_results_df[["xmin", "ymin", "xmax", "ymax"]].values

    # Calculate the mean depth for each bounding box
    depths = extract_depth_within_boxes(output, boxes)

    # Create a list to store the coordinates of detected objects in 3D space
    object_coordinates_3d = []

    # Define the size of the virtual 3D space (adjust as needed)
    virtual_space_size = (1024, 1024, 1024)

    # Loop through detected objects and convert their coordinates
    for j, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        box_depth = depths[j]

        # Calculate the x and y coordinates based on the bounding box
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2

        # Map the depth value to the 3D space (adjust scale factor as needed)
        z = (box_depth - min_depth) / (max_depth - min_depth) * virtual_space_size[2]

        # Append the object's 3D coordinates to the list
        object_coordinates_3d.append({
            "name": object_names[j],
            "x": x,
            "y": y,
            "z": z
        })

    # Save the object coordinates in 3D space as a JSON file
    json_file_path = os.path.join(set_directory, "object_coordinates_3d.json")
    with open(json_file_path, "w") as json_file:
        json.dump(object_coordinates_3d, json_file, indent=4)

    print(f"Processed results for set {i}.")

print("Processing completed.")
