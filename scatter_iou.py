import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def plot_iou_area(predicted_annotation_folder, ground_truth_annotation_folder, image_folder, specified_strings, output_directory):
    area_iou_data = {}  

    for specified_string in specified_strings:
        matching_files = [file for file in os.listdir(ground_truth_annotation_folder) if specified_string in file]
        if len(matching_files) < 1:
            continue

        ground_truth_lines = load_annotations_file([os.path.join(ground_truth_annotation_folder, file) for file in matching_files])
        if len(ground_truth_lines) < 1:
            continue

        ground_truth_annotations = []
        for line in ground_truth_lines:
            class_id = int(line[0])
            points = [(line[i], line[i + 1]) for i in range(1, len(line) - 1, 2)]
            ground_truth_annotations.append((class_id, points))

        for file in tqdm(os.listdir(image_folder)):
            if specified_string in file and file.endswith(".png"):
                image_file_path = os.path.join(image_folder, file)
                predicted_annotation_file = os.path.join(predicted_annotation_folder, file.replace(".png", ".txt"))

                if not os.path.exists(predicted_annotation_file):
                    predicted_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                else:
                    predicted_lines = load_annotations_file([predicted_annotation_file])
                    if len(predicted_lines) < 1:
                        continue

                    predicted_annotations = []
                    for line in predicted_lines:
                        class_id = int(line[0])
                        points = [(line[i], line[i + 1]) for i in range(1, len(line) - 1, 2)]
                        predicted_annotations.append((class_id, points))

                    image = cv2.imread(image_file_path)
                    predicted_masks = create_masks(image.shape, predicted_annotations)

                ground_truth_masks = create_masks(image.shape, ground_truth_annotations)
                iou = calculate_iou(predicted_masks, ground_truth_masks)
                percentage_area = np.sum(predicted_masks) / (image.shape[0] * image.shape[1]) * 100 / 255
                area_iou_data[specified_string] = (percentage_area, iou)

    # Now you can use area_iou_data to plot your scatter plot
    x = [data[0] for data in area_iou_data.values()]
    y = [data[1] for data in area_iou_data.values()]

    plt.scatter(x, y)
    plt.xlabel('Percentage Area')
    plt.ylabel('IoU')
    plt.title('IoU vs Percentage Area')
    plt.savefig(os.path.join(output_directory, 'iou_vs_percentage_area_neww_trial.png'))
    plt.show()

predicted_annotation_folder = '/mnt/vol_b/instance_training_data_segmentation/runs/segment/predict/labels'
ground_truth_annotation_folder = '/mnt/vol_b/instance_training_data_segmentation/labels/val'
image_folder = '/mnt/vol_b/instance_training_data_segmentation/images/val'
specified_strings = ["008_2022-11-03-15-11-30", "012_2022-11-10-16-25-06", "007_2022-11-02-17-35-40",
                      "041_2022-11-17-15-04-53", "044_2022-11-17-15-11-13", "045_2022-11-17-15-13-30",
                      "042_2022-11-17-15-08-53"]
output_directory = '/home/ubuntu/instance_yolov8/'
plot_iou_area(predicted_annotation_folder, ground_truth_annotation_folder, image_folder, specified_strings, output_directory)
