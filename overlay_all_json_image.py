import json
import cv2
import numpy as np
import os

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
    height, width, _ = image.shape

    for annotation in annotations:
        class_id = annotation['category_id']
        segmentation = annotation['segmentation']

        color = (0, 255, 0)
        thickness = 2

        for segment in segmentation:
            points = [(int(x), int(y)) for x, y in zip(segment[0::2], segment[1::2])]
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)

        label_position = tuple(map(int, np.mean(pts, axis=0).flatten()))
        cv2.putText(image, f'Class {class_id}', label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, image)

json_file_path = "/mnt/vol_b/seg_data/final123.json"
output_folder_path = "/mnt/vol_b/overlays"

with open(json_file_path, 'r') as file:
    data = json.load(file)

extracted_data_all_images = extract_info_for_all_files(data)

if extracted_data_all_images['images']:
    

    for image_info in extracted_data_all_images['images']:
        image_file_path = os.path.join("/mnt/vol_b/annot", image_info['file_name'])
        annotations = [ann for ann in extracted_data_all_images['annotations'] if ann['image_id'] == image_info['id']]
        create_overlay(image_file_path, annotations, output_folder_path)

    

