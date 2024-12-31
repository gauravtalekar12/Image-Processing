# import os
# import numpy as np
# from tqdm import tqdm
# import cv2

# def intersect_and_union(pred_mask, gt_mask, num_classes=4, ignore_index=-127):
#     mask = (gt_mask != ignore_index)
#     pred_mask = pred_mask[mask]
#     gt_mask = gt_mask[mask]
#     intersect = pred_mask[pred_mask == gt_mask]

#     area_intersect = np.histogram(intersect, bins=num_classes, range=(0, num_classes-1))[0]
#     # print('ddddddddddddddddd',area_intersect)
#     area_pred_mask = np.histogram(pred_mask, bins=num_classes, range=(0, num_classes-1))[0]
#     area_gt_mask = np.histogram(gt_mask, bins=num_classes, range=(0, num_classes-1))[0]
    
#     # print('ddddddddddddd',area_gt_mask[3])

#     area_union = area_pred_mask + area_gt_mask - area_intersect
#     # print("$$$$$$$$$", area_union[1] > area_intersect[1])
#     return area_intersect, area_union, area_pred_mask, area_gt_mask

# if __name__ == "__main__":
#     pred_mask_dir = "/mnt/vol_c/masks_rug_stable_seaformer"
#     gt_mask_dir = "/mnt/vol_c/rug_gt_mask_final"
    
#     results = []
#     for img_name in tqdm(os.listdir(gt_mask_dir)):
#         gt_mask_path = os.path.join(gt_mask_dir, img_name)
#         pred_mask_path = os.path.join(pred_mask_dir, img_name)
        
#         gt_mask=cv2.imread(gt_mask_path)

#         gt_mask=cv2.resize(gt_mask, (640,640),interpolation=cv2.INTER_NEAREST)//50

#         gt_mask[gt_mask == 2] = 0
#         gt_mask[gt_mask == 3] = 0
#         gt_mask[gt_mask == 4] = 2
#         gt_mask[gt_mask == 5] = 3
#         pred_mask=cv2.imread(pred_mask_path)
#         pred_mask=cv2.resize(pred_mask,(640,640), interpolation=cv2.INTER_NEAREST)//50

#         # print("pred",np.unique(pred_mask),np.unique(gt_mask))
#         results.append(intersect_and_union(pred_mask, gt_mask))
        


#     results = tuple(zip(*results))
    
#     assert len(results) == 4

#     total_area_intersect = sum(results[0])
    
#     total_area_union = sum(results[1])
#     total_area_pred_mask = sum(results[2])
#     total_area_gt_mask = sum(results[3])   

#     iou = total_area_intersect / total_area_union
#     acc = total_area_intersect / total_area_gt_mask

#     print("Total Area Intersect:", total_area_intersect)
#     print("Total Area Union:", total_area_union)
#     print("Total Area Predicted Mask:", total_area_pred_mask)
#     print("Total Area Ground Truth Mask:", total_area_gt_mask)
#     print("IoU:", iou)
#     print("Accuracy:", acc)

    


import os
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

def intersect_and_union(pred_mask, gt_mask, num_classes=4, ignore_index=-127):
    mask = (gt_mask != ignore_index)
    pred_mask = pred_mask[mask]
    gt_mask = gt_mask[mask]
    intersect = pred_mask[pred_mask == gt_mask]
    area_intersect = np.histogram(intersect, bins=num_classes, range=(0, num_classes-1))[0]
    area_pred_mask = np.histogram(pred_mask, bins=num_classes, range=(0, num_classes-1))[0]
    area_gt_mask = np.histogram(gt_mask, bins=num_classes, range=(0, num_classes-1))[0]
    area_union = area_pred_mask + area_gt_mask - area_intersect
    
    iou_value_3 = area_intersect[3] / area_union[3] if area_union[3] > 0 else 0
    return area_intersect, area_union, area_pred_mask, area_gt_mask, iou_value_3

if __name__ == "__main__":
    
    pred_mask_dir = "/mnt/vol_c/masks_rug_stable_seaformer"
    gt_mask_dir = "/mnt/vol_c/rug_gt_mask_final"
    output_dir=""

    results = []
    area_percentage_list = []
    iou_list_value_3 = []
    for img_name in tqdm(os.listdir(gt_mask_dir)):
        gt_mask_path = os.path.join(gt_mask_dir, img_name)
        pred_mask_path = os.path.join(pred_mask_dir, img_name)
        gt_mask = cv2.imread(gt_mask_path)
        gt_mask = cv2.resize(gt_mask, (640, 640), interpolation=cv2.INTER_NEAREST) // 50
        gt_mask[gt_mask == 2] = 0
        gt_mask[gt_mask == 3] = 0
        gt_mask[gt_mask == 4] = 2
        gt_mask[gt_mask == 5] = 3
        pred_mask = cv2.imread(pred_mask_path)
        pred_mask = cv2.resize(pred_mask, (640, 640), interpolation=cv2.INTER_NEAREST) // 50
        
        value_mask = (gt_mask == 3)
        print('ssssssssssss',value_mask)
        percent_area = (np.sum(value_mask) / 640*640)* 100
        area_percentage_list.append(percent_area)
        print('ddddddddddddddddddd', area_percentage_list)
        
        result = intersect_and_union(pred_mask, gt_mask)
        results.append(result[:-1])
        iou_list_value_3.append(result[-1])
    print('ddddddddddddddddddd', area_percentage_list)


    # plt.figure(figsize=(10, 6))
    # plt.hist(iou_list_value_3, bins=10, label='IoU for RUG class')
    # plt.xlabel('IoU Value for rug class ')
    # plt.ylabel('Freq')
    # plt.title('Histogram of IoU Values for Rug class')
    # plt.legend()
    # plt.savefig("IoU Value for rug")

    # plt.figure(figsize=(10, 6))
    # plt.scatter(area_percentage_list, iou_list_value_3, color='blue', alpha=0.5)
    # plt.xlabel('Area Percentage for Rug class')
    # plt.ylabel('IoU for Rug class')
    # plt.title('Area Percentage vs IoU')
    # plt.grid(True)
    # plt.savefig("Area Percentage vs IoU")

