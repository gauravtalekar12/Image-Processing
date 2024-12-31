import json
import cv2
import numpy as np
import os

colors_dict = {
    1: (255, 0, 0),   # Blue
    2: (0, 255, 0),   # Green
    3: (128, 0, 128),  # Purple
    4: (0, 0, 255),    # Red
    5: (0, 255, 255)   # Yellow
}

def extract_info_for_all_files(data):
    extracted_data = {'images': [], 'annotations': []}
    for image_info in data['images']:
        file_name = image_info['file_name']
        image_id = image_info['id']
        annotations_for_image = [{
            'id': annotation['id'],
            'image_id': annotation['image_id'],
            'category_id': annotation['category_id'],
            'segmentation': annotation['segmentation']
        } for annotation in data['annotations'] if annotation['image_id'] == image_id]
        extracted_data['images'].append(image_info)
        extracted_data['annotations'].extend(annotations_for_image)
    return extracted_data

def create_overlay(image_path, annotations, output_folder):
    image = cv2.imread(image_path)

    for annotation in annotations:
        class_id = annotation['category_id']
        segmentation = annotation['segmentation']
        color = colors_dict[class_id]

        for segment in segmentation:
            points = [(int(x), int(y)) for x, y in zip(segment[0::2], segment[1::2])]
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            mask = np.zeros_like(image, dtype=np.uint8)
            cv2.fillPoly(mask, [pts], color=color)
            image = cv2.addWeighted(image, 1, mask, 0.5, 0)

    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, image)

json_file_path = "/mnt/vol_b/seg_data/annotations/instances_Test.json"

output_folder_path = '/mnt/vol_b/seg_data/test'

with open(json_file_path, 'r') as file:
    data = json.load(file)

extracted_data_all_images = extract_info_for_all_files(data)

if extracted_data_all_images['images']:
    for image_info in extracted_data_all_images['images']:
        image_file_path = os.path.join("/mnt/vol_b/stable_diffusion/images/original_images", image_info['file_name'])
        annotations = [ann for ann in extracted_data_all_images['annotations'] if ann['image_id'] == image_info['id']]
        create_overlay(image_file_path, annotations, output_folder_path)
