import os
import numpy as np
from tqdm import tqdm
import cv2

def intersect_and_union(pred_mask, gt_mask, num_classes=3, ignore_index=-127):
    mask = (gt_mask != ignore_index)
    pred_mask = pred_mask[mask]
    gt_mask = gt_mask[mask]
    intersect = pred_mask[pred_mask == gt_mask]

    area_intersect = np.histogram(intersect, bins=num_classes, range=(0, num_classes-1))[0]

    area_pred_mask = np.histogram(pred_mask, bins=num_classes, range=(0, num_classes-1))[0]
    area_gt_mask = np.histogram(gt_mask, bins=num_classes, range=(0, num_classes-1))[0]

    area_union = area_pred_mask + area_gt_mask - area_intersect

    return area_intersect, area_union, area_pred_mask, area_gt_mask

if __name__ == "__main__":
    pred_mask_dir = "/workspace/training_logs_merged/masks"
    gt_mask_dir = "/workspace/MERGED_DATA/Test/masks"
    
    results = []
    for img_name in tqdm(os.listdir(gt_mask_dir)):
        gt_mask_path = os.path.join(gt_mask_dir, img_name)
        pred_mask_path = os.path.join(pred_mask_dir, img_name)
        
        gt_mask=cv2.imread(gt_mask_path)

        gt_mask=cv2.resize(gt_mask, (640,640),interpolation=cv2.INTER_NEAREST)

        # gt_mask[gt_mask == 2] = 0
        # gt_mask[gt_mask == 3] = 0
        # gt_mask[gt_mask == 4] = 2
        # gt_mask[gt_mask == 5] = 3
        pred_mask=cv2.imread(pred_mask_path)
        pred_mask=cv2.resize(pred_mask,(640,640), interpolation=cv2.INTER_NEAREST)

        # print("pred",np.unique(pred_mask),np.unique(gt_mask))
        results.append(intersect_and_union(pred_mask, gt_mask))

        
        
    #     if os.path.exists(pred_mask_path):
    #         gt_mask = gt_mask_path
    #         pred_mask = pred_mask_path
    #         results.append(intersect_and_union(pred_mask, gt_mask))

    results = tuple(zip(*results))
    assert len(results) == 4

    total_area_intersect = sum(results[0])
    # print('ddddddddddddddd',total_area_intersect)
    total_area_union = sum(results[1])
    total_area_pred_mask = sum(results[2])
    total_area_gt_mask = sum(results[3])   

    iou = total_area_intersect / total_area_union
    acc = total_area_intersect / total_area_gt_mask

    print("Total Area Intersect:", total_area_intersect)
    print("Total Area Union:", total_area_union)
    print("Total Area Predicted Mask:", total_area_pred_mask)
    print("Total Area Ground Truth Mask:", total_area_gt_mask)
    print("IoU:", iou)
    print("Accuracy:", acc)