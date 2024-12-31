import cv2
import numpy as np
import os  

def read_json(json_path):
    import json
    with open(json_path, 'r') as file:
        data = json.load(file)

    annotations = []
    for annotation in data['annotations']:
        class_id = annotation['category_id']
        segmentation = annotation['segmentation']

        annotations.append((class_id, segmentation))

    return annotations

def plot(image_path, annotations, output_folder):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    for class_id, segmentation in annotations:
        color = (0, 255, 0)  
        thickness = 2

        for segment in segmentation:
            points = [(int(x), int(y)) for x, y in zip(segment[0::2], segment[1::2])]
            print(points)

            
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)

        
        label_position = tuple(map(int, np.mean(pts, axis=0).flatten()))
        cv2.putText(image, f'Class {class_id}', label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    output_path = os.path.join(output_folder, "output_image_new.png")
    cv2.imwrite(output_path, image)


json_file_path = "/mnt/vol_b/json_new/annotations_small_model/rug_great_5/june-segmentation-drivable-area-1-class-training-32164-sda-01-june-inhouse-t47(17-30-35-r).coco_images_frame_000340_annotation.json"
image_file_path = "/mnt/vol_b/json_new/Images_32k/june-segmentation-drivable-area-1-class-training-32164-sda-01-june-inhouse-t47(17-30-35-r).coco_images_frame_000340.PNG"
output_folder_path = "/mnt/vol_b/json_new/overlayed_img_tests"
annotations = read_json(json_file_path)
plot(image_file_path, annotations, output_folder_path)

