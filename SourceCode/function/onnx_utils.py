import cv2
import numpy as np
import re

def letterbox(img, new_shape=(1024, 1024)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] *r)))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, int(round(dh - 0.1)), int(round(dh + 0.1)),
                             int(round(dw - 0.1)), int(round(dw + 0.1)),
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (dw, dh)

def detect_plates(outputs, ratio, pad, conf_thres=0.5, nms_thres=0.4):
    pred = outputs[0]
    if pred.shape[0] == 1:
        pred = np.squeeze(pred, axis=0)
    pred = pred.T
    dw, dh = pad
    boxes = []
    scores = []
    keypoints = []
    for row in pred:
        x, y, w, h = row[:4]
        conf = row[4]
        if conf < conf_thres:
            continue
        left = int((x - w / 2 - dw) / ratio)
        top = int((y - h / 2 - dh) / ratio)
        width = int(w / ratio)
        height = int(h / ratio)
        boxes.append([left, top, width, height])
        scores.append(float(conf))
        kpts = row[5:]
        pts = []
        for i in range(0, len(kpts), 3):
            kx = (kpts[i] - dw)/ratio
            ky = (kpts[i + 1] - dh)/ratio
            kc = kpts[i + 2]
            pts.append((float(kx), float(ky), float(kc)))
        keypoints.append(pts)
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, nms_thres)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({
                "box": boxes[i],
                "score": scores[i],
                "kpts": keypoints[i]
            })
    return results

def warp_transform(img, pts):
    pts = np.array(pts, dtype=np.float32)

    tl, tr, br, bl = pts

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_w = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_h = int(max(height_left, height_right))

    padding_w = int(max_w * 0.05)
    padding_h = int(max_h * 0.05)
    output_w = max_w + padding_w * 2
    output_h = max_h + padding_h * 2

    dst = np.array([
        [padding_w, padding_h],
        [padding_w + max_w - 1, padding_h],
        [padding_w + max_w - 1, padding_h + max_h - 1],
        [padding_w, padding_h + max_h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img,M,(output_w, output_h))
    return warped

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

def ctc_decode(preds, chars):
    preds_idx = np.argmax(preds, axis=2)[0]
    res = []
    for i in range(len(preds_idx)):
        if preds_idx[i] > 0 and (i == 0 or preds_idx[i] != preds_idx[i-1]):
            res.append(chars[preds_idx[i] - 1])
    return "".join(res)

def clean_text(text):
    if not text:
        return ""
    text = text.upper()
    text = re.sub(r'[^A-Z0-9\n]', '', text)
    return text.strip()