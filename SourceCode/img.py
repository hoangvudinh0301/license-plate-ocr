import os
import cv2
import numpy as np
import onnxruntime as ort
from function.onnx_utils import (load_chars, letterbox,
                                 detect_plate, get_boxes_from_map,
                                 box_to_center, preprocess_rec, ctc_decode)
from postprocessing import clean_text
YOLO_MODEL = "model/best_pl_detecton.onnx"
PP_DET_MODEL = "model/ppocr_det_sim.onnx"
PP_REC_MODEL = "model/ppocr_rec_sim.onnx"
DICT_PATH = "en_dict.txt"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

providers = ["CPUExecutionProvider"]
session_yolo = ort.InferenceSession(YOLO_MODEL, providers=providers)
session_det = ort.InferenceSession(PP_DET_MODEL, providers=providers)
session_rec = ort.InferenceSession(PP_REC_MODEL, providers=providers)
CHARS = load_chars(DICT_PATH)

def process_image(image):
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        img = image.copy()
        img_raw = img.copy()
        img_lb, ratio, pad = letterbox(img_raw)
        blob = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0).astype(np.float32) / 255.0
        outputs = session_yolo.run(None, {session_yolo.get_inputs()[0].name: blob})
        plate_boxes = detect_plate(outputs, img_raw, ratio, pad)
        plate_texts = []
        for box in plate_boxes:

            x1, y1, x2, y2 = map(int, box[:4])
            w, h = x2 - x1, y2 - y1
            h_img, w_img = img_raw.shape[:2]
            pad_w = int(w * 0.05)
            pad_h = int(h * 0.05)
            x1_pad = max(0, x1 - pad_w)
            y1_pad = max(0, y1 - pad_h)
            x2_pad = min(w_img, x2 + pad_w)
            y2_pad = min(h_img, y2 + pad_h)
            crop = img_raw[y1_pad:y2_pad, x1_pad:x2_pad]
            h_p, w_p = crop.shape[:2]
            p_det_in = cv2.resize(crop, (int(np.ceil(w_p/32)*32), int(np.ceil(h_p/32)*32)))
            blob_det = p_det_in.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0
            det_outs = session_det.run(None, {session_det.get_inputs()[0].name: blob_det})[0]
            text_boxes = get_boxes_from_map(det_outs)
            text_boxes = sorted(text_boxes, key=lambda b: (box_to_center(b)[1] // 10, box_to_center(b)[0]))

            full_plate_text = ""
            for box in text_boxes:
                pts = box.astype(np.float32)
                pts[:, 0] *= (w_p / p_det_in.shape[1])
                pts[:, 1] *= (h_p / p_det_in.shape[0])
                pts_on_orig = pts.copy()
                pts_on_orig[:, 0] += x1_pad
                pts_on_orig[:, 1] += y1_pad
                cv2.polylines(img, [pts_on_orig.astype(np.int32)], True, (0, 0, 255), 1)
                bx, by, bw, bh = cv2.boundingRect(pts)
                char_crop = crop[max(0, by):min(h_p, by + bh), max(0, bx):min(w_p, bx + bw)]
                if char_crop.size == 0: continue

                rec_in = preprocess_rec(char_crop)
                if rec_in is not None:
                    preds_rec = session_rec.run(None, {session_rec.get_inputs()[0].name: rec_in})[0]
                    text = ctc_decode(preds_rec, CHARS)
                    full_plate_text += text
                    full_plate_text = clean_text(full_plate_text)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, full_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
            plate_texts.append(full_plate_text)
            output_path = os.path.join(OUTPUT_DIR, "pl_recognition_1.png")
            cv2.imwrite(output_path, img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, "\n".join(plate_texts)