import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import shutil
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    
    
    iou = np.sum(intersection) / np.sum(union)
    return iou

def create_masks(image_shape, annotations):
    masks = np.zeros(image_shape[:2], dtype=np.uint8)

    for class_id, points in annotations:
        if len(points) < 1:
            continue

        points = [(int(x * image_shape[1]), int(y * image_shape[0])) for x, y in points]

        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

        masks = np.maximum(masks, mask)

    return masks

def load_annotations_file(file_paths):
    lines = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            lines.extend([list(map(float, line.strip().split())) for line in f.readlines()])
    return lines

def filter_files_by_string(folder, specified_strings):
    full_paths = [os.path.join(folder, file) for file in os.listdir(folder) if any(specified_string in file for specified_string in specified_strings)]
    return full_paths


def plot_iou_area(predicted_annotation_folder, ground_truth_annotation_folder, image_folder, specified_strings, output_directory):
    
    
    arrays_by_basename = {basename: [] for basename in specified_strings}
    result_data = [] 
    matching_files_gt = filter_files_by_string(ground_truth_annotation_folder, specified_strings)
    matching_files_pred = filter_files_by_string(predicted_annotation_folder, specified_strings)
    matching_image_files_gt = filter_files_by_string(image_folder, specified_strings)
    
    for gt_file_path in tqdm(matching_files_gt):
        ground_truth_lines = load_annotations_file([gt_file_path])
        if len(ground_truth_lines) < 1:
            continue
        
        ground_truth_annotations = []
        for line in ground_truth_lines:
            class_id = int(line[0])
            points = [(line[i], line[i + 1]) for i in range(1, len(line) - 1, 2)]
            ground_truth_annotations.append((class_id, points))
        
        base_name = os.path.basename(gt_file_path)[:-4]
        matching_pred_files = [pred_file for pred_file in matching_files_pred if base_name in pred_file]
        image_file_path = [img_file for img_file in matching_image_files_gt if base_name in img_file]
        
        image = cv2.imread(image_file_path[0])
        if not matching_pred_files:
            predicted_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            predicted_lines = load_annotations_file(matching_pred_files)
            if len(predicted_lines) < 1:
                continue
            
            predicted_annotations = []
            for line in predicted_lines:
                class_id = int(line[0])
                points = [(line[i], line[i + 1]) for i in range(1, len(line) - 1, 2)]
                predicted_annotations.append((class_id, points))
            
            predicted_masks = create_masks(image.shape, predicted_annotations)
        
        ground_truth_masks = create_masks(image.shape, ground_truth_annotations)
        iou = calculate_iou(predicted_masks, ground_truth_masks)
        percentage_area = np.sum((predicted_masks) / (image.shape[0] * image.shape[1])) * 100 / 255
        result_data.append((base_name, percentage_area, iou))
        
          
    
    
    return result_data
   
predicted_annotation_folder = '/mnt/vol_b/instance_training_data_segmentation/runs/segment/predict/labels'
ground_truth_annotation_folder = '/mnt/vol_b/instance_training_data_segmentation/labels/val'  
image_folder = '/mnt/vol_b/instance_training_data_segmentation/images/val'
filtered_output_directory= '/home/instance_yolov8/filtered_images' 
specified_strings = ["008_2022-11-03-15-11-30", "012_2022-11-10-16-25-06", "007_2022-11-02-17-35-40",
                      "041_2022-11-17-15-04-53", "044_2022-11-17-15-11-13", "045_2022-11-17-15-13-30",
                      "042_2022-11-17-15-08-53"]
output_directory = '/home/ubuntu/instance_yolov8/'

result = plot_iou_area(predicted_annotation_folder, ground_truth_annotation_folder, image_folder, specified_strings, output_directory)






for file_name, percentage_area, iou in result:
    if iou < 0.6 and percentage_area > 20:
        source_image_path = os.path.join(image_folder, f"{file_name}.png")  
        output_directory = '/home/ubuntu/instance_yolov8/filtered_images'  
        
        
        
        
        shutil.copy(source_image_path, output_directory)

        
        print(f"File {file_name} copied to {output_directory}")


             
         





