import torch
import os
import cv2
import matplotlib.image


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Load MiDaS model
model_type = "DPT_Large"  # Change to your desired MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# add {i}.png
filenames = [
    "presentation/sdxl/sdxl_image",
    "presentation/photo_sdxl/sdxl_photo_image",
    "presentation/sdxl/reimagine/reimagine_image",
    "presentation/photo_sdxl/reimagine/photo_reimagine_image",
    "presentation/sdxl/canny_conditioning/canny_conditioning_image",
    "presentation/photo_sdxl/canny_conditioning/canny_conditioning_image",
    "presentation/semantic_sd15/upscaled/upscaled_semantic_sd15_image",
    "presentation/semantic_sd15/control_net/cnet_generated_image"   # _{i}_{condition}.png
           ]
for file in filenames:
    # Find the index of the last occurrence of '/'
    last_slash_index = file.rfind('/')

    # If '/' is found, remove everything after it (including '/')
    if last_slash_index != -1:
        directory_path = file[:last_slash_index]
        create_directory(f"{directory_path}/depth_map")
    else:
        print("wrong paths provided")
        create_directory("presentation/temp")
        directory_path = "presentation/temp"

    for i in range(9):
        if "cnet_generated_image" in file:
            for cond in ["Boy", "Girl", "Man", "Women"]:
                filename = f"{file}_{i}_{cond}.png"
                print(filename)
                img = cv2.imread(filename)

                if img is None:
                    print("Error: Failed to load image.")

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                input_batch = transform(img).to(device)

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

                # Save the depth map
                matplotlib.image.imsave(f"{directory_path}/depth_map/depth_image_{i}_{cond}.png", output)
        else:
            filename = f"{file}_{i}.png"
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

            # Save the depth map
            matplotlib.image.imsave(f"{directory_path}/depth_map/depth_image_{i}.png", output)

