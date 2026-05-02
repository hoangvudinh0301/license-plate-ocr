import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

def letterbox(img, new_shape=(640, 640)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] *r)))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, int(round(dh - 0.1)), int(round(dh + 0.1)),
                             int(round(dw - 0.1)), int(round(dw + 0.1)),
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (dw, dh)

def detect_plate(outputs, img, ratio, pad, conf_thres=0.5, nms_thres=0.4):
    pred = outputs[0]
    if pred.shape[0] == 1:
        pred = np.squeeze(pred, axis=0)
    pred = pred.T
    boxes_raw = pred[:, :4]
    scores = np.max(pred[:, 4:], axis=1)
    mask = scores > conf_thres
    boxes_raw = boxes_raw[mask]
    scores = scores[mask]
    if len(boxes_raw) == 0:
        return None
    dw, dh = pad
    all_boxes = []
    for row in boxes_raw:
        x, y, w, h = row
        left = int((x - w / 2 - dw) / ratio)
        top = int((y - h / 2 - dh) /ratio)
        width = int(w / ratio)
        height = int(h / ratio)
        all_boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(all_boxes, scores.tolist(), conf_thres, nms_thres)
    results_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            results_boxes.append(all_boxes[i])
    return results_boxes

def preprocess_rec(img, imgH=48, imgW=320):
    h, w = img.shape[:2]
    ratio = w / float(h)
    new_w = min(int(imgH * ratio), imgW)
    resized = cv2.resize(img, (new_w, imgH))
    padded = np.zeros((imgH, imgW, 3), dtype=np.uint8)
    padded[:, :new_w, :] = resized
    padded = padded.astype(np.float32) / 255.0
    padded = (padded - 0.5) / 0.5
    padded = padded.transpose(2, 0, 1)
    padded = np.expand_dims(padded, axis=0)
    return padded

def load_chars(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    box_list = box.astype(np.intp).tolist()
    offset.AddPath(box_list, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    if len(expanded) == 0:
        return box
    return np.array(expanded[0])

def get_mini_boxes_score(pred, box):
    mask = np.zeros(pred.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [box.astype(np.int32)], 1)
    return cv2.mean(pred, mask=mask)[0]

def box_to_center(box):
    return np.mean(box, axis=0)

def get_boxes_from_map(pred, thresh=0.3, box_thresh=0.5, unclip_ratio=1.5):
    if len(pred.shape) == 4:
        pred = pred[0, 0, :, :]
    mask = (pred > thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = pred.shape
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.float32)
        box_score = get_mini_boxes_score(pred, box)
        if box_score < box_thresh:
            continue
        box = unclip(box, unclip_ratio)
        box[:, 0] = np.clip(box[:, 0], 0, w)
        box[:, 1] = np.clip(box[:, 1], 0, h)
        boxes.append(box.astype(np.int32))
    return boxes

def ctc_decode(preds, chars):
    preds_idx = np.argmax(preds, axis=2)[0]
    res = []
    for i in range(len(preds_idx)):
        if preds_idx[i] > 0 and (i == 0 or preds_idx[i] != preds_idx[i-1]):
            res.append(chars[preds_idx[i] - 1])
    return "".join(res)
