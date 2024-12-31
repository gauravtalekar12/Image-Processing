import cv2
import numpy as np
import os

def read_txt(txt_path):
    annotations = []
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            segmentation = [float(x) for x in parts[1:]]
            annotations.append((class_id, segmentation))
    return annotations

def plot(image_path, annotations, output_folder):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    for class_id, segmentation in annotations:
        color = (0, 255, 0) 
        thickness = 2

        
        points = np.array(segmentation, dtype=np.float32)
        points = points.reshape(-1, 1, 2)
        points[:, :, 0] *= width
        points[:, :, 1] *= height

        
        cv2.polylines(image, [points.astype(np.int32)], isClosed=True, color=color, thickness=thickness)

      
        label_position = tuple(map(int, np.mean(points, axis=0).flatten()))
        cv2.putText(image, f'Class {class_id}', label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    output_path = os.path.join(output_folder, "output_image_new.png")
    cv2.imwrite(output_path, image)


txt_file_path = "/mnt/vol_b/json_new/annotations_small_model/rug_great_5/june-segmentation-drivable-area-1-class-training-32164-sda-01-june-inhouse-t47(17-30-35-r).coco_images_frame_000340_annotation.txt"
image_file_path = "/mnt/vol_b/json_new/Images_32k/june-segmentation-drivable-area-1-class-training-32164-sda-01-june-inhouse-t47(17-30-35-r).coco_images_frame_000340.PNG"
output_folder_path = "/mnt/vol_b/json_new/overlayed_img_tests"
annotations = read_txt(txt_file_path)
plot(image_file_path, annotations, output_folder_path)
