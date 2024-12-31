# import os
# import json
# import cv2
# import numpy as np
# def overlay(
#     img_filename,
#     annotations,
#     output_dir,
#     output_filename,  
#     image_id=1680,
# ):
#     img = cv2.imread(os.path.join("/mnt/vol_b/segmentation_5_class_final/images/", img_filename))
        
#     for annotation in annotations:
#         if annotation["image_id"] == image_id:
#             color = (0, 255, 0) 
#             thickness = 2
           

#             for segmentation in annotation["segmentation"]:
#                 points = [(int(x), int(y)) for x, y in zip(segmentation[0::2], segmentation[1::2])]
#                 pts = np.array(points, np.int32)
#                 pts = pts.reshape((-1, 1, 2))
                
#                 cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
#                 label_position = tuple(map(int, np.mean(pts, axis=0).flatten()))
#                 cv2.putText(img, f'Class {image_id}', label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

#     output_path = os.path.join(output_dir, output_filename)  
#     cv2.imwrite(output_path, img)
# if __name__ == "__main__":

#     input_json_path = "/mnt/vol_b/all_jsons/instances_default2.json"
#     output_dir = "/mnt/vol_c"
    
#     with open(input_json_path, 'r') as file:
#         data = json.load(file)
#         overlay(data["images"][1679]["file_name"], data["annotations"], output_dir, "overlay_1680.jpg")  

import os
import json
import cv2
import numpy as np

def overlay(
    img_filename,
    annotations,
    output_dir,
    output_filename,  
    image_id=1680,
):
    img = cv2.imread(os.path.join("/mnt/vol_b/segmentation_5_class_final/images/", img_filename))
        
    for annotation in annotations:
        if annotation["image_id"] == image_id:
            category_id = annotation["category_id"]
            
            color = (0, 255, 0) 
            thickness = 2
           
            for segmentation in annotation["segmentation"]:
                points = [(int(x), int(y)) for x, y in zip(segmentation[0::2], segmentation[1::2])]
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
                label_position = tuple(map(int, np.mean(pts, axis=0).flatten()))
                cv2.putText(img, f'Class {category_id}', label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    output_path = os.path.join(output_dir, output_filename)  
    cv2.imwrite(output_path, img)

if __name__ == "__main__":
    input_json_path = "/mnt/vol_b/all_jsons/instances_default2.json"
    output_dir = "/mnt/vol_c"
    
    with open(input_json_path, 'r') as file:
        data = json.load(file)
        overlay(data["images"][1679]["file_name"], data["annotations"], output_dir, "overlay_1680.jpg")
