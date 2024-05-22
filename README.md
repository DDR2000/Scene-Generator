# Scene-Generator
#### Results in presentation directory,
#### rename after one run
####
## Scripts running order:
### sdxl.py
#### Provide simple_prompts.txt
#### REQ'S: 9 lines
#### Produces images in sdxl and photo_sdxl 
###
### reimagine_sd2.1.py
#### Reimagines sdxl and photo_sdxl images
#### Produces images in reimagine
###
### canny_conditioning.py (conditioning prompt)
#### negative_prompt = 'low quality, bad quality, sketches'
#### conditioning_prompt = "cooking stake"
#### Conditioning with text and Canny model
#### sdxl and photo_sdxl images
#### Produces images in canny_conditioning
###
### semantic_guidance_sd1.5_upscaler.py
#### semantic prompt = ["dining room",
####                   "family dinner",
####                   "birthday party",
####                   "futuristic"]
#### Utilizing simple_prompt as main prompt
#### Generating SD-1-5 images and conditioning with
#### semantic prompt
#### USE WEIGHTS to increase presence
#### Upscales 512 x 512 images to 2048 x 2048
#### Saves new images in semantic_sd15
###
### controlnet_generator.py
#### Variation list = ["Boy", "Girl", "Man", "Women"]
#### Take semantic_sd15 image
#### Generate Canny image
#### Combine Canny image with prompt to generate new images
###
### midas_depth.py
#### Generates depth maps for each image
#### that will be processed by object detection
####
    "presentation/sdxl/sdxl_image",
    "presentation/photo_sdxl/sdxl_photo_image",
    "presentation/sdxl/reimagine/reimagine_image",
    "presentation/photo_sdxl/reimagine/photo_reimagine_image",
    "presentation/sdxl/canny_conditioning/canny_conditioning_image",
    "presentation/photo_sdxl/canny_conditioning/canny_conditioning_image",
    "presentation/semantic_sd15/upscaled/upscaled_semantic_sd15_image",
    "presentation/semantic_sd15/control_net/cnet_generated_image"
###
### hough.py
#### Change parameters for less or more lines
#### Calculates Canny edges and HoughLines
#### Calculates orientation of the image
#### Can be used on cropped-out objects
#### Save visuals in hough
###
### yolov8.py
###
### retina.py
###
### maskrcnn.py
###
### All object detection models:
#### Detects objects
#### Calculates bounding boxes
#### Spatially aligns boxes and depth maps
#### Extracts depth
#### Returns object coordinates as .json