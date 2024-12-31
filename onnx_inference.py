# import numpy as np
# import cv2
# import onnxruntime as ort
# colors_dict = {
      
#     1: (0, 255, 0),   
#     2: (128, 0, 128),
#     3: (0, 0, 255),   
# }
# onnx_model_path = '/mnt/vol_c/onnx_seaformer_tiny/3_class.onnx'
# ort_session = ort.InferenceSession(onnx_model_path)
# img_dir = '/mnt/vol_c/rug_stable_diff/'


# image_path1 = '/mnt/vol_c/rug_stable_diff/004_2023-05-18-12-27-35_0_camera_1_25890.png'
# image_path2 = '/mnt/vol_c/rug_stable_diff/001_2024-01-12-12-49-47_0_s1726.png'
# input_image1 = cv2.imread(image_path1)
# input_image1 = cv2.cvtColor(input_image1, cv2.COLOR_BGR2RGB).astype(np.float32)  
# input_image2 = cv2.imread(image_path2)
# input_image2 = cv2.cvtColor(input_image2, cv2.COLOR_BGR2RGB).astype(np.float32)  
# dsize = (640, 640)
# img1 = cv2.resize(input_image1, dsize)
# img2 = cv2.resize(input_image2, dsize)
# resized_image1 = cv2.resize(input_image1, dsize) / 255.0
# resized_image2 = cv2.resize(input_image2, dsize) / 255.0
# img_batch = np.stack([resized_image1, resized_image2], axis=0)
# transposed_image = np.transpose(img_batch, (0, 3, 1, 2))
# input_name = ort_session.get_inputs()[0].name
# output_name = ort_session.get_outputs()[0].name
# output = ort_session.run([output_name], {input_name: transposed_image})[0]
# segmentation_masks = output.argmax(axis=1)
# resized_masks = np.zeros((2, 640, 640), dtype=np.uint8)
# for i in range(len(segmentation_masks)):
#     resized_masks[i] = cv2.resize(segmentation_masks[i].astype(np.uint8), (640, 640), interpolation=cv2.INTER_NEAREST)
# color_masks = np.zeros((2, 640, 640, 3), dtype=np.uint8)

# for i in range(len(resized_masks)):
#     for class_id, color in colors_dict.items():
#         color_masks[i][resized_masks[i] == class_id] = color
# color_masks_float32 = color_masks.astype(np.float32)
# overlays = []
# for i, image in enumerate([img1, img2]):
#     overlay = cv2.addWeighted(image, 0.5, color_masks_float32[i], 0.5, 0)
#     overlays.append(overlay)
# for i, overlay in enumerate(overlays):
#     cv2.imwrite(f'overlay_new_{i}.png', overlay)



import os
import numpy as np
import cv2
import onnxruntime as ort

colors_dict = {
    1: (0, 255, 0),
    2: (128, 0, 128),
    3: (0, 0, 255),
}

onnx_model_path = '/mnt/vol_c/onnx_seaformer_tiny/3_class.onnx'
ort_session = ort.InferenceSession(onnx_model_path)
img_dir = '/mnt/vol_c/test/'
output_dir = '/mnt/vol_c/masks_rug_stable_seaformer/'
image_files = os.listdir(img_dir)
# print('aaaaaaaaaa',len(image_files))
mean=[123.675, 116.28, 103.53]
std=[58.395, 57.120000000000005, 57.375]

for i in range(0, len(image_files), 2):
    image_name1 = image_files[i]
    # print('first',image_name1)

    image_name2 = image_files[i + 1]
    # print('second',image_name2)
    image_path1 = os.path.join(img_dir, image_name1)
    image_path2 = os.path.join(img_dir, image_name2)
    input_image1 = cv2.imread(image_path1)
    input_image1 = cv2.cvtColor(input_image1, cv2.COLOR_BGR2RGB).astype(np.float32)
    if input_image1 is None:
        continue  
    input_image2 = cv2.imread(image_path2)
    input_image2 = cv2.cvtColor(input_image2, cv2.COLOR_BGR2RGB).astype(np.float32)
    
    dsize = (640, 640)
    img1 = cv2.resize(input_image1, dsize)
    img2 = cv2.resize(input_image2, dsize)
    if input_image1 is None:
        continue
    
    normalized_img1 = ((img1 - mean) / std).astype(np.float32)
    normalized_img2 = ((img2 - mean) / std).astype(np.float32)
    
    img_batch = np.stack([normalized_img1, normalized_img2], axis=0)
    transposed_image = np.transpose(img_batch, (0, 3, 1, 2))
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    output = ort_session.run([output_name], {input_name: transposed_image})[0]
    
    segmentation_masks = output.argmax(axis=1)
    resized_masks = np.zeros((2, 640, 640), dtype=np.uint8)
    for j in range(len(segmentation_masks)):
        resized_masks[j] = cv2.resize(segmentation_masks[j].astype(np.uint8), (640, 640), interpolation=cv2.INTER_NEAREST)

    mask_name1 = os.path.splitext(image_name1)[0] + '.png'
    mask_name2 = os.path.splitext(image_name2)[0] + '.png'
    mask_path1 = os.path.join(output_dir, mask_name1)
    mask_path2 = os.path.join(output_dir, mask_name2)
    # print('ddddddddddddd',np.unique(resized_masks[0]*50))
    
    cv2.imwrite(mask_path1, resized_masks[0]*50)
    cv2.imwrite(mask_path2, resized_masks[1]*50)





    # color_masks = np.zeros((2, 640, 640, 3), dtype=np.uint8)
    
    # for j in range(len(resized_masks)):
    #     for class_id, color in colors_dict.items():
    #         color_masks[j][resized_masks[j] == class_id] = color
    
    # color_masks_float32 = color_masks.astype(np.float32)
    # overlays = []
    
    # for j, image in enumerate([img1, img2]):
    #     overlay = cv2.addWeighted(image, 0.5, color_masks_float32[j], 0.5, 0)
    #     overlays.append(overlay)
    
    # for j, overlay in enumerate(overlays):
    #     output_path = os.path.join(output_dir, f'overlay_new_{image_name1}_{image_name2}.png')
    #     cv2.imwrite(output_path, overlay)


