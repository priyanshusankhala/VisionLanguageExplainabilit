from typing import Union, List, Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def dict2box(d: Dict):
    return d['x'], d['y'], d['w'], d['h']


def plot_bbox(image, x: Union[int, List] = None, y: Union[int, List] = None, w: Union[int, List] = None,
              h: Union[int, List] = None, boxes: Union[List, Dict] = None, color: Tuple[int, int, int] = (255, 0, 0),
              border: int = 2, texts: List[str] = None, show: bool = False, title: str = None):
    if isinstance(image, Image.Image):
        image = np.float32(image)

    if boxes is not None:
        if isinstance(boxes, list):
            if isinstance(boxes[0], (tuple, list)):
                assert len(boxes[0]) == 4, "List of boxes must contain tuple (x, y, w, h)."
                _boxes = boxes
            elif isinstance(boxes[0], dict):
                assert {'x', 'y', 'w', 'h'}.difference(
                    set(boxes[0].keys())) == set(), "Dict of boxes must contain keys 'x', 'y', 'w', 'h'."
                _boxes = [dict2box(obj) for obj in boxes]
            else:
                raise ValueError('Unknown box format')
        elif isinstance(boxes, dict):
            assert {'x', 'y', 'w', 'h'}.difference(
                set(boxes.keys())) == set(), "Dict of boxes must contain keys 'x', 'y', 'w', 'h'."
            if isinstance(boxes['x'], list):
                _boxes = zip(boxes['x'], boxes['y'], boxes['w'], boxes['h'])
            elif isinstance(boxes['x'], int):
                _boxes = [dict2box(boxes)]
            else:
                raise ValueError()
        else:
            raise ValueError('Boxes must be List or Dict.')
    elif all(v is not None for v in [x, y, w, h]):
        x = [x] if isinstance(x, int) else x
        y = [y] if isinstance(y, int) else y
        w = [w] if isinstance(w, int) else w
        h = [h] if isinstance(h, int) else h
        _boxes = zip(x, y, w, h)
    else:
        raise ValueError("No boxes found.")
    for i, (_x, _y, _w, _h) in enumerate(_boxes):
        image = cv2.rectangle(image, (_x, _y), (_x + _w, _y + _h), color, border)
        if texts is not None:
            cv2.putText(image, texts[i], (_x, _y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

    if show:
        plt.imshow(image / 255)
        plt.axis('off')
        if title:
            plt.title(title)
        plt.show()
    return image


def computeIoU(box1, box2):
    if isinstance(box1, dict):
        box1 = dict2box(box1)
    if isinstance(box2, dict):
        box2 = dict2box(box2)
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0] + box1[2] - 1, box2[0] + box2[2] - 1)
    inter_y2 = min(box1[1] + box1[3] - 1, box2[1] + box2[3] - 1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return float(inter) / union
