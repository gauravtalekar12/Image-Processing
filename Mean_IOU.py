import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def calculate_iou(mask1, mask2):
    
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    return intersection,union

def create_masks(image_shape, annotations):
    masks = np.zeros(image_shape[:2], dtype=np.uint8)

    for class_id, points in annotations:
        if len(points)<1:
            continue
        
        points = [(int(x * image_shape[1]), int(y * image_shape[0])) for x, y in points]

        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

        masks = np.maximum(masks, mask)

    return masks

def load_annotations_file(file_path):
    with open(file_path, 'r') as f:
        lines = [list(map(float, line.strip().split())) for line in f.readlines()]
        
        
    return lines

def compute_mean_iou(predicted_annotation_folder, ground_truth_annotation_folder, image_folder):
    ground_truth_files = os.listdir(ground_truth_annotation_folder)
    total_intersection = 0
    total_union=0

    for ground_truth_file in tqdm(ground_truth_files):
        if ground_truth_file.endswith(".txt"):
            ground_truth_annotation_file = os.path.join(ground_truth_annotation_folder, ground_truth_file)

            ground_truth_lines = load_annotations_file(ground_truth_annotation_file)
            #print(ground_truth_lines)
            #breakpoint()
            if len(ground_truth_lines) < 1:
                continue

            ground_truth_annotations = []

            for line in ground_truth_lines:
                class_id = int(line[0])
                points = [(line[i], line[i + 1]) for i in range(1, len(line) - 1, 2)]
                ground_truth_annotations.append((class_id, points))
                #print(ground_truth_annotations)
                #breakpoint()

            image_file_path = os.path.join(image_folder, ground_truth_file.replace(".txt", ".png"))
            predicted_annotation_file = os.path.join(predicted_annotation_folder, ground_truth_file)
            #print(predicted_annotation_file)
            #breakpoint()

            if not os.path.exists(predicted_annotation_file):
                
                predicted_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            else:
                predicted_lines = load_annotations_file(predicted_annotation_file)
                #print(predicted_lines)
                #breakpoint()

                if len(predicted_lines) < 1:
                    continue

                predicted_annotations = []

                for line in predicted_lines:
                    class_id = int(line[0])
                    points = [(line[i], line[i + 1]) for i in range(1, len(line) - 1, 2)]
                    predicted_annotations.append((class_id, points))

                image = cv2.imread(image_file_path)
                #breakpoint()
                predicted_masks = create_masks(image.shape, predicted_annotations)
                plt.imshow(predicted_masks, cmap='gray')
                plt.title('Predicted  Mask')
                plt.axis('off')
                plt.show()
                output_directory = '/home/ubuntu/instance_yolov8/'
                cv2.imwrite(os.path.join(output_directory, 'predicted_masks_test.png'), predicted_masks)
                
                # breakpoint()

            ground_truth_masks = create_masks(image.shape, ground_truth_annotations)

            intersection, union = calculate_iou(predicted_masks, ground_truth_masks)
            total_intersection += np.sum(intersection)
            total_union+=np.sum(union)
            

    mean_iou = total_intersection / total_union
    return mean_iou


predicted_annotation_folder = '/mnt/vol_b/instance_training_data_segmentation/runs/segment/predict/labels'
ground_truth_annotation_folder = '/mnt/vol_b/instance_training_data_segmentation/labels/val'
image_folder = '/mnt/vol_b/instance_training_data_segmentation/images/val'

mean_iou = compute_mean_iou(predicted_annotation_folder, ground_truth_annotation_folder, image_folder)

print(f"Mean IoU: {mean_iou}")
