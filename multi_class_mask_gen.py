import os
import cv2
import numpy as np
from tqdm import tqdm

def read_yolo_format(txt_path, image_width, image_height):
    polygons = []
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])  
            coordinates = [float(parts[i]) * image_width if i % 2 == 1 else float(parts[i]) * image_height for i in range(1, len(parts))]
            points = np.array(coordinates, dtype=np.int32).reshape((-1, 2))
            polygons.append((class_id, points))
    return polygons

def draw_polygons(mask, polygons):
    for class_id, points in polygons:
        if len(points) > 0:
            if class_id == 0:  
                color = 0
            elif class_id == 1:
                color = 0
            elif class_id == 2: 
                color = 1
            else:
                color = 0  
            
            cv2.fillPoly(mask, [points], color=color)
    return mask

if __name__ == "__main__":
   
    gt_txt_dir = '/mnt/vol_d/5_class_instance_seg/labels/train'
    image_dir = '/mnt/vol_c/rug_images'
    output_dir = '/mnt/vol_c/rug_masks'  
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for img_name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, img_name)
        gt_txt_path = os.path.join(gt_txt_dir, img_name[:-4] + ".txt")
    
        image_height, image_width = 640, 640
        gt_mask = np.zeros((640, 640), dtype=np.uint8)
        pred_mask = np.zeros((640, 640), dtype=np.uint8)
        
        if os.path.exists(gt_txt_path):
            gt_polygons = read_yolo_format(gt_txt_path, image_width, image_height)
            gt_mask = draw_polygons(gt_mask, gt_polygons)
    
    
        
        
        cv2.imwrite(os.path.join(output_dir, img_name[:-4] + "_mask.png"), gt_mask * 255)
