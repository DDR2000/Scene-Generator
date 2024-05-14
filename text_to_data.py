from diffusers.utils import load_image
import torch
import os
import cv2
import matplotlib.image
from diffusers import StableDiffusionXLPipeline
from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers import SemanticStableDiffusionPipeline
from diffusers import StableDiffusionUpscalePipeline
import json
from ultralytics import YOLO
import sys
from torchvision.models.detection import retinanet
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL


def read_prompts(filename):
    try:
        with open(filename, 'r') as file:
            prompt_text = file.read()
            print("Prompt read successfully:")
            print(prompt_text)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")


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

# Load MiDaS model
model_type = "DPT_Large"  # Change to your desired MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")


if len(sys.argv) != 2:
    print("Usage: python script.py prompts.txt")
else:
    prompt_file = sys.argv[1]
    read_prompts(prompt_file)

    # Read prompts from a file
    with open(prompt_file, "r") as file:
        prompts = file.read().splitlines()


# Function to extract depth within bounding boxes
def extract_depth_within_boxes(depth_map, objects_info):
    depths = []
    for obj_idx, obj in objects_info.items():
        print(f"processing object {obj_idx}")
        xmin, ymin, xmax, ymax = map(int, obj["box"])  # Convert to integers
        box_depth = depth_map[ymin:ymax, xmin:xmax].mean()
        depths.append(box_depth)
    return depths


def run_yolo(set_directory):
    # Detect objects and bounding boxes using YOLO
    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")
    img = [f"{set_directory}/image.png"]
    results = model(img)
    # Save YOLOv8 detection results as images
    boxes = results[0].boxes  # Boxes object for bounding box outputs
    # masks = results[0].masks  # Masks object for segmentation masks outputs
    # keypoints = results[0].keypoints  # Keypoints object for pose outputs
    # probs = results[0].probs  # Probs object for classification outputs
    # results[0].show()  # display to screen

    # Make directory to save yolo results
    create_directory(f"{set_directory}/yolo_result")

    results[0].save(filename=f"{set_directory}/yolo_result/result.jpg")

    objects_info = {}
    obj_cnt = 0
    for box in boxes:
        objects_info[str(obj_cnt)] = {"class": model.names[int(box.cls)], "box": box.xyxy.cpu().numpy()[0].tolist()}
        obj_cnt += 1

    return objects_info


def run_retina_net(set_directory, threshold=0.5):
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
    # Load image
    img = Image.open(f"{set_directory}/image.png")
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
    for i, box in enumerate(boxes_c):
        obj_info = {
            "class": COCO_INSTANCE_CATEGORY_NAMES[labels_c[i]],
            "box": box.tolist()
        }
        return_retina_net_objects_info[str(i)] = obj_info

    # Visualize
    img_with_detections = img.copy()
    plt.imshow(img_with_detections)

    ax = plt.gca()
    retina_net_objects_info = list(zip(boxes, labels, scores))
    for box, label, score in retina_net_objects_info:
        x1, y1, x2, y2 = box
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
        category = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
        plt.text(x1, y1, f'{category} {score:.2f}', bbox=dict(facecolor='red', alpha=0.5), fontsize=12, color='white')
    plt.axis('off')

    # Remove white frame
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Set figure size to maintain 1024x1024 resolution
    plt.gcf().set_size_inches(1024 / 100, 1024 / 100)

    # Make directory to save retina_net results
    create_directory(f"{set_directory}/retina_net_result")
    result_img_path = f"{set_directory}/retina_net_result/image_with_detections.png"
    plt.savefig(result_img_path, dpi=100, bbox_inches='tight', pad_inches=0)  # Save the current figure with detections
    plt.close()  # Close the figure to release resources

    return return_retina_net_objects_info


def run_maskrcnn(set_directory, threshold=0.5):
    # Load the pre-trained Mask R-CNN model
    maskrcnn_model = maskrcnn_resnet50_fpn(pretrained=True)

    # Set the model to evaluation mode
    maskrcnn_model.eval()

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

    # Load image
    img = Image.open(f"{set_directory}/image.png")
    # Resize image to 1024x1024
    #img = img.resize((1024, 1024))

    img_tensor = F.to_tensor(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        prediction = maskrcnn_model(img_tensor)

    # Filter out detections below threshold
    boxes = prediction[0]['boxes'][prediction[0]['scores'] > threshold]
    labels = prediction[0]['labels'][prediction[0]['scores'] > threshold]
    masks = prediction[0]['masks'][prediction[0]['scores'] > threshold]

    # Convert PyTorch tensors to NumPy arrays
    boxes_c = boxes.cpu().numpy()
    labels_c = labels.cpu().numpy()
    # masks_c = masks.cpu().numpy()

    # Organize detection information in a dictionary
    maskrcnn_objects_info = {}
    for i, box in enumerate(boxes_c):
        obj_info = {
            "class": COCO_INSTANCE_CATEGORY_NAMES[labels_c[i]],
            "box": box.tolist()
        }
        maskrcnn_objects_info[str(i)] = obj_info

    # Visualize
    img_with_detections = img.copy()
    img_np = np.array(img_with_detections)

    ax = plt.gca()
    for box, label, mask in zip(boxes, labels, masks):
        x1, y1, x2, y2 = box
        category = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
        plt.text(x1, y1, f'{category}', bbox=dict(facecolor='red', alpha=0.5), fontsize=12, color='white')
        mask = mask[0].mul(255).byte().cpu().numpy()
        img_np[mask == 255] = (img_np[mask == 255] * 0.5) + (np.array([255, 0, 0]) * 0.5)
    plt.imshow(img_np)
    plt.axis('off')

    # Remove white frame
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Set figure size to maintain 1024x1024 resolution
    plt.gcf().set_size_inches(1024 / 100, 1024 / 100)

    # Make directory to save retina_net results
    create_directory(f"{set_directory}/maskrcnn_result")
    result_img_path = f"{set_directory}/maskrcnn_result/image_with_detections.png"
    plt.savefig(result_img_path, bbox_inches='tight', pad_inches=0)  # Save the current figure with detections , dpi=100, bbox_inches='tight', pad_inches=0
    plt.close()  # Close the figure to release resources

    return maskrcnn_objects_info


def run_hough_transform(image_path):  # Try dotted line, or transparent lines
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

        # Show the result
        cv2.imshow('Image with Hough Lines', hough_image)
        # Save image to disk
        create_directory(f"{set_directory}/hough_lines_result")
        result_img_path = f"{set_directory}/hough_lines_result/image_with_hough_lines.png"
        cv2.imwrite(result_img_path, hough_image)
        print(f"Result image saved to: {result_img_path}")

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    else:
        print("No lines detected.")
        orientation = None

    return orientation


def reimagine(image_path):  # unclip (reimagine)
    reimagine_pipe = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip",
                                                                 torch_dtype=torch.float16, variation="fp16")
    reimagine_pipe.to("cuda")

    # get image

    init_image = load_image(image_path)


    # run image variation
    images = reimagine_pipe(init_image).images

    create_directory(f"{set_directory}/reimagine")
    images[0].save(f"{set_directory}/reimagine/unclip_image_0.png")


def control_net_conditioning(image_path, conditioning_prompt):
    negative_prompt = 'low quality, bad quality, sketches'
    init_image = load_image(image_path)
    controlnet_conditioning_scale = 0.5  # recommended for good generalization

    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    conditioning_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
    )
    # pipe.inference_mode()
    conditioning_pipe.enable_model_cpu_offload()

    cond_image = np.array(init_image)
    cond_image = cv2.Canny(cond_image, 100, 200)
    cond_image = cond_image[:, :, None]
    cond_image = np.concatenate([cond_image, cond_image, cond_image], axis=2)
    cond_image = Image.fromarray(cond_image)
    cond_image.show()
    images = conditioning_pipe(
        conditioning_prompt, negative_prompt=negative_prompt, image=cond_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images

    images[0].show()
    create_directory(f"{set_directory}/conditioning_result")
    result_img_path = f"{set_directory}/conditioning_result/conditioned_image.png"
    result_image_np = np.array(images[0])
    cv2.imwrite(result_img_path, result_image_np)


def semantic_guidance(small_prompt, editing_prompt):
    semantic_pipe = SemanticStableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    )
    semantic_pipe = semantic_pipe.to("cuda")

    out = semantic_pipe(
        prompt=small_prompt,
        num_images_per_prompt=1,
        guidance_scale=7,
        editing_prompt=editing_prompt,
        reverse_editing_direction=[
            False,
            False,
            False,
            False,
        ],  # Direction of guidance i.e. increase all concepts
        edit_warmup_steps=[10, 10, 10, 10],  # Warmup period for each concept
        edit_guidance_scale=[4, 5, 5, 5.4],  # Guidance scale for each concept
        edit_threshold=[
            0.99,
            0.975,
            0.925,
            0.96,
        ],
        # Threshold for each concept. Threshold equals the percentile of the latent space that will be discarded.
        # I.e. threshold=0.99 uses 1% of the latent dimensions
        edit_momentum_scale=0.3,  # Momentum scale that will be added to the latent guidance
        edit_mom_beta=0.6,  # Momentum beta
        edit_weights=[1, 1, 1, 1, 1],  # Weights of the individual concepts against each other
    )
    semantic_image = out.images[0]
    semantic_image.show()
    torch.cuda.empty_cache()
    upscaler(semantic_image, small_prompt)


def upscaler(low_res_img, simple_prompt):
    # load model and scheduler
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")
    upscaled_image = pipeline(prompt=simple_prompt, image=low_res_img).images[0]
    create_directory(f"{set_directory}/upscaled_images")
    result_img_path = f"{set_directory}/upscaled_images/upscaled_image.png"
    upscaled_image.save(result_img_path)
    return


# Process each set of prompts
for i, prompt in enumerate(prompts):
    # Create a directory for the current set of results
    set_directory = os.path.join(results_directory, f"results_{i}")
    create_directory(set_directory)

    # Prompt addition
    prompt = prompt + "Normal Angle, Mirrorless, 55mm, f/1.8, shutter 1/30, iso 10000"

    # Generate an image using Stable Diffusion XL
    image = pipe(prompt).images[0]
    image.save(f"{set_directory}/image.png")

    filename = f"{set_directory}/image.png"

    # Generate reimagine
    #reimagine(filename)
    #torch.cuda.empty_cache()
    # Generate control net conditioning image
    #conditioning_prompt = "student doing homework on computer"
    #control_net_conditioning(filename, conditioning_prompt)

    # Semantic guidance image generation using SD_V1.5 + Upscaler
    #simple_prompt = "office"
    #semantic_prompt = ["desk",
    #                   "chair",
    #                   "computer",
    #                   "flower pot"]

    #semantic_guidance(simple_prompt, semantic_prompt)

    # Find object orientation, find orientation of extracted objects
    #predominant_orientation = run_hough_transform(filename)
    #print("predominant_orientation: ", predominant_orientation)

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

    # Run yolo
    objects_info = run_yolo(set_directory)
    print("yolo_objects_info")
    print(objects_info)

    # Run RetinaNet
    #retina_objects_info = run_retina_net(set_directory, threshold=0.5)
    #print("retina_objects_info")
    #print(retina_objects_info)

    # Run MaskRCNN
    #maskrcnn_objects_info = run_maskrcnn(set_directory, threshold=0.5)
    #print("maskrcnn_objects_info")
    #print(maskrcnn_objects_info)

    # Calculate the mean depth for each bounding box
    depths = extract_depth_within_boxes(output, objects_info)

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
    json_file_path = os.path.join(set_directory, "object_coordinates_3d.json")
    with open(json_file_path, "w") as json_file:
        json.dump(object_coordinates_3d, json_file, indent=4)

    print(f"Processed results for set {i}.")

print("Processing completed.")


