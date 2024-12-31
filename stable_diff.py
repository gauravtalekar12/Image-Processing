
from diffusers import StableDiffusionInpaintPipeline
import torch
import cv2
import numpy as np
import random
from PIL import Image
import glob
import os
device = "cuda"
# pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16,).to(device)

pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float32).to(device)
# pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16,).to(device)

prompts = [
    "Generate a Persian-style woolen rug with traditional motifs",
    "Generate a luxurious silk rug with intricate borders",
    "Generate a bamboo rug with a minimalist aesthetic",
    "Create a sisal rug with coastal inspired pattern",
    "Design a jute rug with natural textures",
    "Put woolen Rug",
    "Generate a vintage style leather-rug with embossed details"
]

def process_image(im_pth, ms_pth, output_folder):
    img = Image.open(im_pth).convert("RGB")
    mask = cv2.imread(ms_pth)
    mask = cv2.resize(mask, (640, 640))
    mask = Image.fromarray(mask.astype('uint8'))
    img = img.resize((640, 640))
    prompt = random.choice(prompts)

    transformed_image = pipe(prompt=prompt, image=img, mask_image=mask,height=640,width=640).images[0]


    filename = os.path.basename(im_pth)
    output_path = os.path.join(output_folder, filename)
    transformed_image.save(output_path)

def process_all_images(image_folder, mask_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(image_folder):
            
            im_pth = os.path.join(image_folder, filename)
            ms_pth = os.path.join(mask_folder, filename.replace(".png", "_mask.png"))

            if os.path.exists(ms_pth):
                process_image(im_pth, ms_pth, output_folder)

image_folder = "/mnt/vol_c/rug_imgs"
mask_folder = "/mnt/vol_c/rug_mask_new"
output_folder = "/mnt/vol_c/test"

process_all_images(image_folder, mask_folder, output_folder)

