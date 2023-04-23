import os
import json

dir = './data/' + 'test/'

ann_folder = dir + 'labels'

img_folder = dir + 'images'

if not os.path.exists(dir + 'json'):
    os.makedirs(dir + 'json')

for ann_file in os.listdir(ann_folder):
    img_name = os.path.splitext(ann_file)[0]

    with open(os.path.join(ann_folder, ann_file), 'r') as f:
        ann_data = f.read().splitlines()

    objects = []
    for ann_line in ann_data:
        parts = ann_line.split()

        x, y, w, h = map(int, parts[1:])

        objects.append({
            'bbox': [x, y, w, h],
            'category_id': int(parts[0])
        })

    json_data = {
        'file_name': os.path.join(img_folder, f'{img_name}.jpg'),
        'image_id': img_name,
        'annotations': objects
    }
    with open(os.path.join(dir + 'json', f'{img_name}.json'), 'w') as f:
        json.dump(json_data, f)

















# {
#     "file_name": "path/to/image.jpg",
#     "image_id": "image001",
#     "annotations": [
#         {
#             "bbox": [x1, y1, w1, h1],
#             "category_id": 0
#         },
#         {
#             "bbox": [x2, y2, w2, h2],
#             "category_id": 1
#         },
#         {
#             "bbox": [x3, y3, w3, h3],
#             "category_id": 2
#         },
#         ...
#     ]
# }