# scripts/convert_bdd100k_to_yolo.py
import os
import json
from tqdm import tqdm

def convert(json_path, output_dir, image_folder):
    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, 'r') as f:
        data = json.load(f)

    for item in tqdm(data):
        image_name = item['name']
        image_path = os.path.join(image_folder, image_name)
        if not os.path.exists(image_path):
            continue  # skip if image file doesn't exist
        label_path = os.path.join(output_dir, image_name.replace('.jpg', '.txt'))
        with open(label_path, 'w') as f_out:
            for label in item['labels']:
                if label['category'] != 'pedestrian':
                    continue

                box2d = label['box2d']
                x1, y1, x2, y2 = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']

                x_center = (x1 + x2) / 2 / 1280  # BDD100K is usually 1280x720
                y_center = (y1 + y2) / 2 / 720
                width = (x2 - x1) / 1280
                height = (y2 - y1) / 720

                f_out.write(f"0 {x_center} {y_center} {width} {height}\n")


if __name__ == '__main__':
    base = 'dataset/bdd100k'

    convert(
        json_path=f'{base}/labels/bdd100k_labels_images_train.json',
        output_dir=f'{base}/labels/train',
        image_folder=f'{base}/images/100k/train'
    )

    convert(
        json_path=f'{base}/labels/bdd100k_labels_images_val.json',
        output_dir=f'{base}/labels/val',
        image_folder=f'{base}/images/100k/val'
    )
