import json
from pycocotools.coco import COCO

def merge_coco_json(json_files, output_file):
    merged_annotations = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": [],
        
    }

    image_id = 0
    annotation_id= 0
    category_id = 0
    existing_category_ids = set()

    for idx, file in enumerate(json_files):
        coco = COCO(file)

        for image in coco.dataset['images']:
            image['id'] += image_id
            merged_annotations['images'].append(image)

        for annotation in coco.dataset['annotations']:
            annotation['id'] += annotation_id
            annotation['image_id'] += image_id
            merged_annotations['annotations'].append(annotation)

        for category in coco.dataset['categories']:
            if category['id'] not in existing_category_ids:
                category['id'] += category_id
                merged_annotations['categories'].append(category)
                existing_category_ids.add(category['id'])

        image_id = len(merged_annotations['images'])
        annotation_id = len(merged_annotations['annotations'])
        category_id = len(merged_annotations['categories'])

    with open(output_file, 'w') as f:
        json.dump(merged_annotations, f)

json_files = ["/mnt/vol_b/segmentation_5_class_data/annotations_10k/006_2024-01-17-13-25-36_0/annotations/instances_default.json", "/mnt/vol_b/segmentation_5_class_data/annotations_10k/pap3_images1/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_10k/pap3_images2/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_10k/pap3_images3/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_10k/pap3_images4/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_10k/pap3_images5/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/000_2023-05-18-15-19-19_0_camera_0/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/001_2023-06-16-12-06-58_0_camera_1/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/003_2023-04-27-11-07-55_0camera_0/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/003_2023-06-21-11-21-33_0_camera_1/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/008_2023-05-04-11-01-54_0_camera_1/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/014_2023-09-06-11-24-01_0camera_1/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/000_2023-05-19-09-49-05_0_camera_0/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/002_2023-05-12-14-22-05_0_camera_1/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/003_2023-05-18-12-25-26_0_camera_0/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/005_2023-06-15-15-08-33_0_camera_1/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/009_2023-05-04-14-28-25_0_camera_0/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/015_2023-09-06-11-30-37_0camera_1/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/000_2023-06-01-15-08-26_0_camera_1/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/002_2023-06-02-11-11-04_0_camera_1/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/003_2023-05-18-12-25-26_0_camera_1/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/006_2023-04-28-08-18-06_0camera_0/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/011_2023-05-05-16-02-46_0_camera_0/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/001_2023-06-02-11-10-30_0_camera_0/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/002_2023-06-21-11-19-02_0_camera_0/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/003_2023-06-13-16-12-11_0_camera_1/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/007_2023-05-04-09-48-07_0_camera_0/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/011_2023-05-05-16-02-46_0_camera_1/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/001_2023-06-02-11-10-30_0_camera_1/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/002_2023-06-21-11-19-02_0_camera_1/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/003_2023-06-21-11-21-33_0_camera_0/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/007_2023-05-04-09-48-07_0_camera_1/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_7083/014_2023-09-06-11-24-01_0camera_0/annotations/instances_default.json","/mnt/vol_b/segmentation_5_class_data/annotations_760/rug_updated_annotations/annotations/instances_Test.json"]

output_file = "merged_files_final.json"
merge_coco_json(json_files, output_file)

