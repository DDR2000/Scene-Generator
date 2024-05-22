import os
import cv2
import json
from ultralytics import YOLO
import numpy as np
from PIL import Image


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


model = YOLO("yolov8n.pt")

def run_yolo(image_path):
    # Detect objects and bounding boxes using YOLO
    # Load YOLOv8 model

    img = [image_path]
    results = model(img)
    # Save YOLOv8 detection results as images
    boxes = results[0].boxes  # Boxes object for bounding box outputs
    # masks = results[0].masks  # Masks object for segmentation masks outputs
    # keypoints = results[0].keypoints  # Keypoints object for pose outputs
    # probs = results[0].probs  # Probs object for classification outputs
    # results[0].show()  # display to screen

    # Make directory to save yolo results


    objects_info = {}
    obj_cnt = 0
    for box in boxes:
        objects_info[str(obj_cnt)] = {"class": model.names[int(box.cls)], "box": box.xyxy.cpu().numpy()[0].tolist()}
        obj_cnt += 1

    return results, objects_info


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
        yolo_path = f"{directory_path}/yolo"
        create_directory(yolo_path)
    else:
        # Dump in temp
        print("wrong paths provided")
        create_directory("presentation/temp")
        yolo_path = "presentation/temp"
        directory_path = yolo_path

    for i in range(9):
        if "cnet_generated_image" in file:
            for cond in ["Boy", "Girl", "Man", "Women"]:
                filename = f"{file}_{i}_{cond}.png"
                results, objects_info = run_yolo(filename)
                results[0].save(filename=f"{yolo_path}/yolo_image_{i}_{cond}.jpg")

                # Load depth map image into numpy array.
                depth_map_image_path = f"{directory_path}/depth_map/depth_image_{i}_{cond}.png"
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
                json_file_path = f"{yolo_path}/object_coordinates_3d_{i}_{cond}.json"
                with open(json_file_path, "w") as json_file:
                    json.dump(object_coordinates_3d, json_file, indent=4)

                print(f"Processed results for set {i}_{cond} in {directory_path}")

        else:
            filename = f"{file}_{i}.png"

            results, objects_info = run_yolo(filename)
            results[0].save(filename=f"{yolo_path}/yolo_image_{i}.jpg")

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
            json_file_path = f"{yolo_path}/object_coordinates_3d_{i}.json"
            with open(json_file_path, "w") as json_file:
                json.dump(object_coordinates_3d, json_file, indent=4)

            print(f"Processed results for set {i} in {directory_path}")

print("Processing completed.")


