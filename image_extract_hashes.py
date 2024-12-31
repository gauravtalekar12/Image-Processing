import json
import os
import shutil
from multiprocessing import Pool, cpu_count

def process_image(image_info):
    hash_value, image_path, base_path, output_folder = image_info
    source_path = os.path.join(base_path, image_path)
    output_path = os.path.join(output_folder, os.path.basename(image_path))

    try:
        os.makedirs(output_folder, exist_ok=True)
        shutil.copy(source_path, output_path)
        print(f"Image '{image_path}' copied to '{output_path}'")
    except Exception as e:
        print(f"Error copying image '{image_path}': {str(e)}")

def extract_images_from_json(json_path, output_folder, base_path):
    with open(json_path, "r") as f:
        hashes = json.load(f)

    image_infos = []

    for hash_value, paths in hashes.items():
        for image_path in paths:
            image_infos.append((hash_value, image_path, base_path, output_folder))

    with Pool(cpu_count()) as pool:
        pool.map(process_image, image_infos)

if __name__ == "__main__":
    images_base_path = "/mnt/vol_b/segmentation_data/merged_dataset_1_class/images"
    output_folder = "/mnt/vol_b/json_new/Images_32k"
    json_path = "/mnt/vol_b/json_new/remaining_images.json"
    extract_images_from_json(json_path, output_folder, images_base_path)
