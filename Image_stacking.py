# import os
# import shutil
# import numpy as np
# import cv2
# from tqdm import tqdm

# def intersect_and_union(pred_mask, gt_mask, num_classes=4, ignore_index=-127):
#     mask = (gt_mask != ignore_index)
#     pred_mask = pred_mask[mask]
#     gt_mask = gt_mask[mask]
#     intersect = pred_mask[pred_mask == gt_mask]
#     area_intersect = np.histogram(intersect, bins=num_classes, range=(0, num_classes-1))[0]
#     area_pred_mask = np.histogram(pred_mask, bins=num_classes, range=(0, num_classes-1))[0]
#     area_gt_mask = np.histogram(gt_mask, bins=num_classes, range=(0, num_classes-1))[0]
#     area_union = area_pred_mask + area_gt_mask - area_intersect
#     return area_intersect, area_union, area_pred_mask, area_gt_mask
# org_image,
# def compute_metrics(results):
#     results = tuple(zip(*results))
#     total_area_intersect = np.sum(results[0], axis=0)
#     total_area_union = np.sum(results[1], axis=0)
#     iou = total_area_intersect / total_area_union
#     return iou

# if __name__ == "__main__":
#     pred_mask_dir = "/mnt/vol_c/5000_masks"
#     gt_mask_dir = "/mnt/vol_c/5000_gt_mask"
#     output_dir = "/mnt/vol_c/mY_test"
#     image_dir = "/mnt/vol_c/5000_images"
#     original_image_dir = "/mnt/vol_c/original_5000"
#     bin_mask_dir="/mnt/vol_c/bin_5000"

    

#     results = []
#     for img_name in tqdm(os.listdir(gt_mask_dir)):
#         gt_mask_path = os.path.join(gt_mask_dir, img_name)
#         pred_mask_path = os.path.join(pred_mask_dir, img_name)
#         image_path = os.path.join(image_dir, img_name)
#         org_image=os.path.join(original_image_dir, img_name)
#         bin_mask=os.path.join(bin_mask_dir, img_name)
        

        
#         gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
#         gt_mask = cv2.resize(gt_mask, (640, 640), interpolation=cv2.INTER_NEAREST) // 50
#         gt_mask[gt_mask == 2] = 0
#         gt_mask[gt_mask == 3] = 0
#         gt_mask[gt_mask == 4] = 2
#         gt_mask[gt_mask == 5] = 3

#         pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
#         pred_mask = cv2.resize(pred_mask, (640, 640), interpolation=cv2.INTER_NEAREST) // 50

#         iou_results = intersect_and_union(pred_mask, gt_mask)
#         results.append(iou_results)

#         iou = compute_metrics([iou_results])
#         iou_str = f"{iou[-1]:.4f}".split('.')[1]
#         shutil.copy(image_path, os.path.join(output_dir, f"{iou_str}_stable_diffusion_{img_name}"))


# import os
# import shutil
# import numpy as np
# import cv2
# from tqdm import tqdm
# from multiprocessing import Pool, cpu_count

# def intersect_and_union(args):
#     pred_mask, gt_mask = args
#     mask = (gt_mask != ignore_index)
#     pred_mask = pred_mask[mask]
#     gt_mask = gt_mask[mask]
#     intersect = pred_mask[pred_mask == gt_mask]
#     area_intersect = np.histogram(intersect, bins=num_classes, range=(0, num_classes-1))[0]
#     area_pred_mask = np.histogram(pred_mask, bins=num_classes, range=(0, num_classes-1))[0]
#     area_gt_mask = np.histogram(gt_mask, bins=num_classes, range=(0, num_classes-1))[0]
#     area_union = area_pred_mask + area_gt_mask - area_intersect
#     return area_intersect, area_union, area_pred_mask, area_gt_mask

# def compute_metrics(results):
#     results = tuple(zip(*results))
#     total_area_intersect = np.sum(results[0], axis=0)
#     total_area_union = np.sum(results[1], axis=0)
#     iou = total_area_intersect / total_area_union
#     return iou

# def process_image(img_name):
#     original_image_path = os.path.join(original_image_dir, img_name)
#     image_path = os.path.join(image_dir, img_name)
#     gt_mask_path = os.path.join(gt_mask_dir, img_name)
#     pred_mask_path = os.path.join(pred_mask_dir, img_name)

#     original_image = cv2.imread(original_image_path)
#     original_image = cv2.resize(original_image, (640, 640))
#     print('dddddddddddddddd',original_image.shape)

#     image = cv2.imread(image_path)
#     print('ssssssssssssss',image.shape)
#     image = cv2.resize(image, (640, 640))

#     gt_mask = cv2.imread(gt_mask_path)
#     gt_mask = cv2.resize(gt_mask, (640, 640), interpolation=cv2.INTER_NEAREST)//50

#     binary_mask = np.zeros_like(gt_mask, dtype=np.uint8) 
#     binary_mask[gt_mask == 3] = 255  
#     print('abababababababa',binary_mask.shape)
#     binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
#     print('ssssssssssssssssss',binary_mask.shape)

#     pred_mask = cv2.imread(pred_mask_path)
#     pred_mask = cv2.resize(pred_mask, (640, 640), interpolation=cv2.INTER_NEAREST)//50
#     print('aaaaaaaaaaaaa',gt_mask.shape)
#     print('bbbbbbbbbbbbbbbb',pred_mask.shape)
#     print('cccccccccccccccccccc',pred_mask.shape)

#     iou_results = intersect_and_union((pred_mask, gt_mask))
#     iou = compute_metrics([iou_results])
#     print('kkkkkkkkkkkkkkkkkkkkkkkkk',iou)
#     print('ssssssssssssss',iou)
#     iou_str = f"{iou[-1]:.4f}".split('.')[1]

#     stacked_image = np.hstack((original_image, image, binary_mask))
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1
#     font_color = (255, 0, 0)
#     thickness = 2
#     cv2.putText(stacked_image, "Original Image", (10, 30), font, font_scale, font_color, thickness)
#     cv2.putText(stacked_image, "Generated Image", (650, 30), font, font_scale, font_color, thickness)
#     cv2.putText(stacked_image, "Binary Mask", (1290, 30), font, font_scale, font_color, thickness)
#     cv2.imwrite(os.path.join(output_dir, f"{iou_str}_stacked_{img_name}"), stacked_image)


# if __name__ == "__main__":
#     pred_mask_dir = "/mnt/vol_c/5000_masks"
#     gt_mask_dir = "/mnt/vol_c/5000_gt_mask"
#     output_dir = "/mnt/vol_c/mY_test"
#     image_dir = "/mnt/vol_c/5000_images"
#     original_image_dir = "/mnt/vol_c/original_5000"

#     ignore_index = -127
#     num_classes = 4

#     pool = Pool(cpu_count())
#     images = os.listdir(gt_mask_dir)
#     for _ in tqdm(pool.imap_unordered(process_image, images), total=len(images)):
#         pass
#     pool.close()
#     pool.join()

# import os
# import shutil
# import numpy as np
# import cv2
# from tqdm import tqdm

# def intersect_and_union(pred_mask, gt_mask, num_classes=4, ignore_index=-127):
#     mask = (gt_mask != ignore_index)
#     pred_mask = pred_mask[mask]
#     gt_mask = gt_mask[mask]
#     intersect = pred_mask[pred_mask == gt_mask]
#     area_intersect = np.histogram(intersect, bins=num_classes, range=(0, num_classes-1))[0]
#     area_pred_mask = np.histogram(pred_mask, bins=num_classes, range=(0, num_classes-1))[0]
#     area_gt_mask = np.histogram(gt_mask, bins=num_classes, range=(0, num_classes-1))[0]
#     area_union = area_pred_mask + area_gt_mask - area_intersect
#     return area_intersect, area_union, area_pred_mask, area_gt_mask

# def compute_metrics(results):
#     results = tuple(zip(*results))
#     total_area_intersect = np.sum(results[0], axis=0)
#     total_area_union = np.sum(results[1], axis=0)
#     iou = total_area_intersect / total_area_union
#     return iou

# if __name__ == "__main__":
#     pred_mask_dir = "/mnt/vol_c/masks_rug_stable_seaformer"
#     gt_mask_dir = "/mnt/vol_c/rug_gt_mask_final"
#     output_dir = "/mnt/vol_c/stacked_images_rug"
#     image_dir = "/mnt/vol_c/stable_diffusion_rug_gen"
#     original_image_dir = "/mnt/vol_c/original_rug_images"
#     bin_mask_dir="/mnt/vol_c/binary_masks_rug"

#     results = []
#     for img_name in tqdm(os.listdir(gt_mask_dir)):
#         gt_mask_path = os.path.join(gt_mask_dir, img_name)
#         pred_mask_path = os.path.join(pred_mask_dir, img_name)
#         image_path = os.path.join(image_dir, img_name)
#         org_image=os.path.join(original_image_dir, img_name)
#         bin_mask=os.path.join(bin_mask_dir, img_name)
        
#         gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
#         gt_mask = cv2.resize(gt_mask, (640, 640), interpolation=cv2.INTER_NEAREST) // 50
#         gt_mask[gt_mask == 2] = 0
#         gt_mask[gt_mask == 3] = 0
#         gt_mask[gt_mask == 4] = 2
#         gt_mask[gt_mask == 5] = 3

#         pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
#         pred_mask = cv2.resize(pred_mask, (640, 640), interpolation=cv2.INTER_NEAREST) // 50

#         iou_results = intersect_and_union(pred_mask, gt_mask)
#         results.append(iou_results)

#         iou = compute_metrics([iou_results])
#         iou_str = f"{iou[-1]:.4f}".split('.')[1]
#         org_img = cv2.imread(org_image)
#         img = cv2.imread(image_path)
#         bin_mask_img = cv2.imread(bin_mask)
    
#         org_img_resized = cv2.resize(org_img, (img.shape[1], img.shape[0]))
#         bin_mask_img_resized = cv2.resize(bin_mask_img, (img.shape[1], img.shape[0]))
#         stacked_img = np.hstack([org_img_resized, img, bin_mask_img_resized])
#         output_filename = f"{iou_str}_stable_diffusion_{img_name}"
#         cv2.imwrite(os.path.join(output_dir, output_filename), stacked_img)

import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool

def intersect_and_union(pred_mask, gt_mask, num_classes=4, ignore_index=-127):
    mask = (gt_mask != ignore_index)
    pred_mask = pred_mask[mask]
    gt_mask = gt_mask[mask]
    intersect = pred_mask[pred_mask == gt_mask]
    area_intersect = np.histogram(intersect, bins=num_classes, range=(0, num_classes-1))[0]
    area_pred_mask = np.histogram(pred_mask, bins=num_classes, range=(0, num_classes-1))[0]
    area_gt_mask = np.histogram(gt_mask, bins=num_classes, range=(0, num_classes-1))[0]
    area_union = area_pred_mask + area_gt_mask - area_intersect
    return area_intersect, area_union, area_pred_mask, area_gt_mask

def compute_metrics(results):
    results = tuple(zip(*results))
    total_area_intersect = np.sum(results[0], axis=0)
    total_area_union = np.sum(results[1], axis=0)
    iou = total_area_intersect / total_area_union
    return iou

# def process_image(img_name):
#     gt_mask_path = os.path.join(gt_mask_dir, img_name)
#     pred_mask_path = os.path.join(pred_mask_dir, img_name)
#     image_path = os.path.join(image_dir, img_name)
#     org_image = os.path.join(original_image_dir, img_name)
#     bin_mask = os.path.join(bin_mask_dir, img_name)
    
#     gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
#     gt_mask = cv2.resize(gt_mask, (640, 640), interpolation=cv2.INTER_NEAREST) // 50
#     gt_mask[gt_mask == 2] = 0
#     gt_mask[gt_mask == 3] = 0
#     gt_mask[gt_mask == 4] = 2
#     gt_mask[gt_mask == 5] = 3

#     pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
#     pred_mask = cv2.resize(pred_mask, (640, 640), interpolation=cv2.INTER_NEAREST) // 50

#     iou_results = intersect_and_union(pred_mask, gt_mask)
#     # print('gggggggggggggggggggg',iou_results)

#     iou = compute_metrics([iou_results])
#     total_area = 640 * 640
    

#     if iou[-1] < 0.8:
#         iou_str = f"{iou[-1]:.4f}".split('.')[1]
#         org_img = cv2.imread(org_image)
#         img = cv2.imread(image_path)
#         bin_mask_img = cv2.imread(bin_mask)
        
#         org_img_resized = cv2.resize(org_img, (img.shape[1], img.shape[0]))
#         bin_mask_img_resized = cv2.resize(bin_mask_img, (img.shape[1], img.shape[0]))
#         stacked_img = np.hstack([org_img_resized, img, bin_mask_img_resized])
#         output_filename = f"{iou_str}_stable_diffusion_{img_name}"
#         cv2.imwrite(os.path.join(iou_filt, output_filename), stacked_img)


# if __name__ == "__main__":
#     pred_mask_dir = "/mnt/vol_c/masks_rug_stable_seaformer"
#     gt_mask_dir = "/mnt/vol_c/rug_gt_mask_final"
#     output_dir = "/mnt/vol_c/stacked_images_rug"
#     image_dir = "/mnt/vol_c/stable_diffusion_rug_gen"
#     original_image_dir = "/mnt/vol_c/original_rug_images"
#     bin_mask_dir="/mnt/vol_c/binary_masks_rug"
#     iou_filt="/mnt/vol_c/iou_less_60_percent_stable_diffusion"

#     img_names = os.listdir(gt_mask_dir)
    
#     with Pool(processes=4) as pool:
#         list(tqdm(pool.imap(process_image, img_names), total=len(img_names)))
def process_image(img_name):
    gt_mask_path = os.path.join(gt_mask_dir, img_name)
    pred_mask_path = os.path.join(pred_mask_dir, img_name)
    image_path = os.path.join(image_dir, img_name)
    org_image = os.path.join(original_image_dir, img_name)
    bin_mask = os.path.join(bin_mask_dir, img_name)
    
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.resize(gt_mask, (640, 640), interpolation=cv2.INTER_NEAREST) // 50
    gt_mask[gt_mask == 2] = 0
    gt_mask[gt_mask == 3] = 0
    gt_mask[gt_mask == 4] = 2
    gt_mask[gt_mask == 5] = 3

    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.resize(pred_mask, (640, 640), interpolation=cv2.INTER_NEAREST) // 50

    iou_results = intersect_and_union(pred_mask, gt_mask)
    iou = compute_metrics([iou_results])
    total_area = 640 * 640
    
    if iou[-1] < 0.8:
        iou_str = f"{iou[-1]:.4f}".split('.')[1]
        org_img = cv2.imread(org_image)
        img = cv2.imread(image_path)
        bin_mask_img = cv2.imread(bin_mask)
        
        org_img_resized = cv2.resize(org_img, (img.shape[1], img.shape[0]))
        bin_mask_img_resized = cv2.resize(bin_mask_img, (img.shape[1], img.shape[0]))
        
       
        output_org_filename = f"{iou_str}_stable_diffusion_{img_name}"
        cv2.imwrite(os.path.join(output_org_dir, output_org_filename), img)
        
        output_bin_mask_filename = f"{iou_str}_stable_diffusion_{img_name}"
        cv2.imwrite(os.path.join(output_bin_mask_dir, output_bin_mask_filename), bin_mask_img_resized)
if __name__ == "__main__":
    pred_mask_dir = "/mnt/vol_c/masks_rug_stable_seaformer"
    gt_mask_dir = "/mnt/vol_c/rug_gt_mask_final"
    output_dir = "/mnt/vol_c/stacked_images_rug"
    image_dir = "/mnt/vol_c/stable_diffusion_rug_gen"
    original_image_dir = "/mnt/vol_c/original_rug_images"
    bin_mask_dir="/mnt/vol_c/binary_masks_rug"
    iou_filt="/mnt/vol_c/iou_less_60_percent_stable_diffusion"
    output_org_dir = "/mnt/vol_c/stable_diff_0.8_images"
    output_bin_mask_dir = "/mnt/vol_c/stable_diff_0.8_masks"

    img_names = os.listdir(gt_mask_dir)
    
    with Pool(processes=4) as pool:
        list(tqdm(pool.imap(process_image, img_names), total=len(img_names)))