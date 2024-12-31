# import cv2
# import numpy as np
# file_path = '/mnt/vol_c/5_class_new_data/masks/High-Resolution-d-Data#Aug-High-Resolution-d-13-c-Object-d-t-24308-Inhouse##th16-001_2022-08-18-13-29-46_right##TH16-001_2022-08-18-13-29-46_right_frame_1491.png'
# img = cv2.imread(file_path)
# mask=np.where(img==250,255,0).astype(np.uint8)
# output_file_path = '/mnt/vol_c/mask_output.png'
# cv2.imwrite(output_file_path, mask)

import cv2
import os
import multiprocessing
import numpy as np

def process_image(file_path, output_dir):
    img = cv2.imread(file_path)
    
    if np.any(img == 250):
        mask = np.where(img == 250, 255, 0).astype(np.uint8)
        
        output_file_name = os.path.basename(file_path)
        output_file_path = os.path.join(output_dir, output_file_name)
        cv2.imwrite(output_file_path, mask)


def main():
    input_dir = "/mnt/vol_c/5_class_new_data/masks"
    output_dir = "/mnt/vol_c/new_bin"
    files = [os.path.join(input_dir, file) for file in os.listdir(input_dir)]

    pool = multiprocessing.Pool()
    pool.starmap(process_image, [(file, output_dir) for file in files])
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()






