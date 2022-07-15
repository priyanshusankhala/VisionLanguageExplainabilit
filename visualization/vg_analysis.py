import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from nltk.corpus import wordnet as wn
from skimage import transform as skimage_transform
from tqdm import tqdm

from bbox import plot_bbox, computeIoU
from tools import visualize

gradcam_root = Path('/home/ts/Downloads/s/gradcam_clip')

with open('/home/ts/Documents/VisualGenome/objects.json', 'r') as f:
    vg_objects = json.load(f)


def get_vg_objects(_id):
    return list(filter(lambda x: x['image_id'] == int(_id), vg_objects))[0]['objects']


def load_image(_id):
    return Image.open(f'/home/ts/Documents/VisualGenome/images/{_id}.jpg').convert('RGB')


def load_gradcam(_id):
    with open(gradcam_root / f'gradcam_{_id}.json', 'r') as file:
        gradcam = json.load(file)
    return gradcam


all_image_ids = list(map(lambda x: x.stem.split('_')[1], gradcam_root.iterdir()))
image_id_to_size = {}
for i in all_image_ids:
    image = load_image(i)
    image_id_to_size[i] = image.size


def evaluate(image_ids,
             tactic='samples',  # weighted_blob, largest_blob, samples
             block_id=2,  # 0-5 or mean
             grad_thresh=0.15, alpha=0.5,
             debug=False):
    total = 0
    pos = 0
    classwise = defaultdict(lambda: dict(total=0, pos=0))

    for i, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
        gradcam = load_gradcam(image_id)
        objects = get_vg_objects(image_id)

        if debug:
            image = load_image(image_id)
            plot_bbox(image, boxes=objects, title='VG boxes', show=True)

        for obj, cam in gradcam.items():
            synset = set(map(lambda x: x.name(), wn.synsets(obj, pos=wn.NOUN)))
            gt = list(filter(lambda x: synset.intersection(set(x['synsets'])), objects))
            if not gt:
                continue

            cam = np.array(list(map(lambda x: x['gradcam'], cam)))  # shape: features, blocks, 24, 24
            print(cam.shape)
            cam = cam.mean(axis=0)  # mean over all features
            cam = cam[0]

            # if block_id == 'mean':
            #     cam = cam.mean(axis=0)
            # else:
            #     cam = cam[block_id]  # choose block

            cam = cam.reshape(7, 7)

            cam = skimage_transform.resize(cam, (image_id_to_size[image_id][::-1]), order=3, mode='constant')
            pred_box = None

            if tactic == 'samples':
                max_score = 0
                for det in objects:
                    score = cam[int(det['y']):int(det['y'] + det['h']), int(det['x']):int(det['x'] + det['w'])]
                    area = det['w'] * det['h']
                    score = score.sum() / area ** alpha
                    if score > max_score:
                        pred_box = det
                        max_score = score
            elif tactic == 'largest_blob':
                cam_thresh = np.where(cam > grad_thresh * cam.max(), 1.0, 0.0).astype(np.uint8)
                contours, hierarchy = cv2.findContours(cam_thresh, 1, 2)
                contours = max(contours, key=lambda x: len(x))
                pred_box = cv2.boundingRect(contours)
                if debug:
                    plot_bbox(cam_thresh.copy() * 255, boxes=[pred_box], show=True, color=(255, 0, 0))
            elif tactic == 'weighted_blob':
                cam_thresh = np.where(cam > grad_thresh * cam.max(), 1.0, 0.0).astype(np.uint8)
                contours, hierarchy = cv2.findContours(cam_thresh, 1, 2)

                max_score = 0
                best_box = None

                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    score = cam[int(y):int(y + h), int(x):int(x + w)]
                    area = w * h
                    score = score.sum() / area ** alpha

                    if score > max_score:
                        best_box = (x, y, w, h)
                        max_score = score
                pred_box = best_box
                if debug:
                    plot_bbox(cam_thresh.copy() * 255, boxes=[pred_box], texts=[str(round(max_score, 3))], show=True,
                              color=(255, 0, 0))

            iou = computeIoU(pred_box, gt[0])
            iou = [computeIoU(pred_box, gt[i]) for i in range(len(gt))]
            iou = max(iou)

            total += 1
            classwise[obj]['total'] += 1
            if iou >= 0.5:
                pos += 1
                classwise[obj]['pos'] += 1

            if debug:
                image_box = plot_bbox(image, boxes=[pred_box])
                image_box = plot_bbox(image_box, boxes=gt, color=(0, 255, 0))
                visualize(image_box, cam, title=obj)

    accuracy = pos / total
    classwise = sorted(list(map(lambda x: (x[0], x[1]['pos'] / x[1]['total']), classwise.items())), key=lambda x: x[1],
                       reverse=True)
    if debug:
        print("Total Accuracy:", accuracy)
        for cls, val in classwise:
            print(f"{cls : <20} {val : .2f}")

    return accuracy, classwise


acc, cls_acc = evaluate(image_ids=['2350672'],
                        tactic='samples',  # weighted_blob, largest_blob, samples
                        block_id=2,  # 0-5 or mean
                        grad_thresh=0.1,
                        alpha=0.7,
                        debug=True)
