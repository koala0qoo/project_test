import os
import numpy as np
import logging
from lxml import etree
from object_detection.utils import dataset_util

def CAMmap(feature_maps, predictions, n_top):
    map_size = feature_maps.shape[0:2]
    heatmap = np.zeros((map_size[0], map_size[1], n_top))
    tops = np.argsort(-predictions)
    for i in range(n_top):
        feature_map = feature_maps[:, :, tops[i]]
        #heatmap[:, :, i] = (feature_map - feature_map.min())/(feature_map.max() - feature_map.min())
        heatmap[:, :, i] = feature_map

    return heatmap


def bounding_box(heatmap, threshold):
    n_boxes = heatmap.shape[2]
    map_size = heatmap.shape[0]
    boxes = np.zeros((n_boxes, 4))
    for i in range(n_boxes):
        ymin, xmin, ymax, xmax = 0, 0, 1, 1
        x1, x2, y1, y2 = False, False, False, False
        for j in range(map_size):
            if x1 == True & y1 == True:
                break
            for k in range(map_size):
                if (heatmap[j, k, i] >= threshold) & (y1 == False):
                    ymin = j / map_size
                    y1 = True
                if (heatmap[k, j, i] >= threshold) & (x1 == False):
                    xmin = j / map_size
                    x1 = True
        for j in reversed(range(map_size)):
            if x2 == True & y2 == True:
                break
            for k in range(map_size):
                if (heatmap[j, k, i] >= threshold) & (y2 == False):
                    ymax = (j + 1) / map_size
                    y2 = True
                if (heatmap[k, j, i] >= threshold) & (x2 == False):
                    xmax = (j + 1) / map_size
                    x2 = True
        bbox = [ymin, xmin, ymax, xmax]
        boxes[i, :] = bbox

    return boxes


def grey2rainbow(grey):
    grey = 255 - grey
    h, w = grey.shape
    rainbow = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if grey[i, j] <= 51:
                rainbow[i, j, 0] = 255
                rainbow[i, j, 1] = grey[i, j] * 5
                rainbow[i, j, 2] = 0
            elif grey[i, j] <= 102:
                rainbow[i, j, 0] = 255 - (grey[i, j] - 51) * 5
                rainbow[i, j, 1] = 255
                rainbow[i, j, 2] = 0
            elif grey[i, j] <= 153:
                rainbow[i, j, 0] = 0
                rainbow[i, j, 1] = 255
                rainbow[i, j, 2] = (grey[i, j] - 102) * 5
            elif grey[i, j] <= 204:
                rainbow[i, j, 0] = 0
                rainbow[i, j, 1] = 255 - int((grey[i, j] - 153) * 128 / 51 + 0.5)
                rainbow[i, j, 2] = 255
            elif grey[i, j] <= 255:
                rainbow[i, j, 0] = 0
                rainbow[i, j, 1] = 127 - int((grey[i, j] - 204) * 127 / 51 + 0.5)
                rainbow[i, j, 2] = 255

    return rainbow


def bilinear(img, h, w):
    height, width, channels = img.shape
    if h == height and w == width:
        return img
    new_img = np.zeros((h, w, channels), np.uint8)
    scale_x = float(width) / w
    scale_y = float(height) / h
    for n in range(channels):
        for dst_y in range(h):
            for dst_x in range(w):
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                src_x_0 = int(np.floor(src_x))
                src_y_0 = int(np.floor(src_y))
                src_x_1 = min(src_x_0 + 1, width - 1)
                src_y_1 = min(src_y_0 + 1, height - 1)

                value0 = (src_x_1 - src_x) * img[src_y_0, src_x_0, n] + (src_x - src_x_0) * img[src_y_0, src_x_1, n]
                value1 = (src_x_1 - src_x) * img[src_y_1, src_x_0, n] + (src_x - src_x_0) * img[src_y_1, src_x_1, n]
                new_img[dst_y, dst_x, n] = int((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1)
    return new_img


def get_boxes(annotations_dir, image_id):
    boxes = []
    classes = []
    xml_path = os.path.join(annotations_dir, 'xmls', 'test_{0}.xml'.format(image_id))
    if not os.path.exists(xml_path):
        logging.warning('Could not find %s, ignoring example.', xml_path)
        return boxes, classes

    xml_str = open(xml_path, 'r').read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    if 'object' in data:
        for obj in data['object']:

            xmin = float(obj['bndbox']['xmin'])
            xmax = float(obj['bndbox']['xmax'])
            ymin = float(obj['bndbox']['ymin'])
            ymax = float(obj['bndbox']['ymax'])

            box = [ymin, xmin, ymax, xmax]
            boxes.append(box)

            class_label = int(obj['name'])
            classes.append(class_label)

    return boxes, classes
