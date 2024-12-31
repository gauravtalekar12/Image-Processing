import json
import os
import numpy as np
import cv2
from pycocotools import mask as mask_utils
from tqdm import tqdm

def create_coco_annotation(image_path, annotation_path, output_dir):
    coco_data = {
        "licenses": [{"name": "rug", "id": 0, "url": ""}],
        "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
        "categories": [
            {"id": 1, "name": "rug", "supercategory": ""}
        ],
        "images": [],
        "annotations": []
    }

    annotation_id = 1

    image_id = 1

    image = cv2.imread(image_path)
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    annotation = cv2.imread(annotation_path, 0)
    # if np.any(annotation == 27):
    #     print(True)

    coco_data["images"].append({
        "id": image_id,
        "width": image.shape[1],
        "height": image.shape[0],
        "file_name": os.path.basename(image_path),
        "license": 0,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": 0,
    })

    binary_mask = np.zeros_like(annotation, dtype=np.uint8)
    binary_mask[annotation == 28] = 1

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    segmentation = []
    for contour in contours:
        if cv2.contourArea(contour) < 50:
            continue
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        contour_new = approx.flatten().tolist()

        if len(contour_new) < 8 or len(contour_new) % 2 != 0:
            continue
        segmentation.append(contour_new)

    if segmentation:
        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": segmentation,
            "area": float(np.sum(binary_mask)),
            "bbox": cv2.boundingRect(binary_mask),
            "iscrowd": 0,
        })

        json_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_annotation.json"
        json_path = os.path.join(output_dir, json_filename)

        with open(json_path, "w") as json_file:
            json.dump(coco_data, json_file, indent=4)

output_directory = "/mnt/vol_b/json_new/annotations_small_model/rug_great_5"
image_directory = "/mnt/vol_b/json_new/rug_great_5_new"
annotation_directory = "/mnt/vol_b/json_new/final_pred"            


for filename in tqdm(os.listdir(image_directory),desc="Processing images"):
    
    image_path = os.path.join(image_directory, filename)
    annotation_path = os.path.join(annotation_directory, filename)
    create_coco_annotation(image_path, annotation_path, output_directory)