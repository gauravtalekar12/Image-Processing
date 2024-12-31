import cv2
import os

mask_folder = '/mnt/vol_c/5000_masks'
image_folder = '/mnt/vol_c/sorted_5000'
output_folder = '/mnt/vol_c/5000_overlayed'

for img_file in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_file)
    if os.path.isfile(img_path):
        common_name = img_file[22:]  # Extracts a common identifier to match images and masks
        for mask_file in os.listdir(mask_folder):
            mask_path = os.path.join(mask_folder, mask_file)
            if os.path.isfile(mask_path) and common_name in mask_file:
                image = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)


                if image.shape[:2] != mask.shape[:2]:
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        
                if len(image.shape) == 3 and len(mask.shape) == 2:  # Image is color and mask is grayscale
                    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        
                try:
                    overlay = cv2.addWeighted(image, 1, mask, 0.5, 0)
                    output_path = os.path.join(output_folder, img_file)
                    cv2.imwrite(output_path, overlay)
                except cv2.error as e:
                    print(f"Failed to overlay image and mask: {e}")    