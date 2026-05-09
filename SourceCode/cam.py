import os

import numpy as np
import onnxruntime as ort
import cv2
import imageio

from preprocessing import get_plate_corners, perspective_transform, preprocess_plate
from postprocessing import process_ocr_results, extract_with_score, stable_results, clean_text
from function.onnx_utils import (load_chars, letterbox,
                                 detect_plate, get_boxes_from_map,
                                 box_to_center, preprocess_rec, ctc_decode)
import supervision as sv

YOLO_MODEL = "model/best_pl_detecton.onnx"
PP_DET_MODEL = "model/ppocr_det_sim.onnx"
PP_REC_MODEL = "model/ppocr_rec_sim.onnx"
DICT_PATH = "en_dict.txt"
VIDEO_PATH = "samples/IMG_6917.MOV"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

providers = ["CPUExecutionProvider"]
session_yolo = ort.InferenceSession(YOLO_MODEL, providers=providers)
session_det = ort.InferenceSession(PP_DET_MODEL, providers=providers)
session_rec = ort.InferenceSession(PP_REC_MODEL, providers=providers)
CHARS = load_chars(DICT_PATH)

def process_video(video_path):
    tracker = sv.ByteTrack()
    results_cache = {}

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_path = "results/output_result.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    detections = sv.Detections.empty()
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % 5 == 0:
            img_raw = frame.copy()
            img_lb, ratio, pad = letterbox(img_raw)
            blob = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
            blob = blob.transpose(2, 0, 1)
            blob = np.expand_dims(blob, axis=0).astype(np.float32) / 255.0
            outputs = session_yolo.run(None, {session_yolo.get_inputs()[0].name: blob})
            raw_boxes = detect_plate(outputs, img_raw, ratio, pad)

            if raw_boxes is not None and len(raw_boxes) > 0:
                detections = sv.Detections(
                    xyxy=np.array([b[:4] for b in raw_boxes], dtype=np.float32),
                    confidence=np.array([b[4] for b in raw_boxes], dtype=np.float32),
                    class_id=np.zeros(len(raw_boxes), dtype=int)
                )
                detections = tracker.update_with_detections(detections)
            else: detections = sv.Detections.empty()

        for i in range(len(detections)):
            coords = detections.xyxy[i].astype(int)
            conf_now = detections.confidence[i]
            track_id = detections.tracker_id[i] if detections.tracker_id is not None else None
            x1, y1, x2, y2 = coords
            w, h = x2 - x1, y2 - y1
            cached_data = results_cache.get(track_id, {"text": "", "score": 0.0})
            full_plate_text = cached_data["text"]
            if not full_plate_text or conf_now > cached_data["score"]:
                h_img, w_img = img_raw.shape[:2]
                pad_w = int(w * 0.05)
                pad_h = int(h * 0.05)
                x1_pad = max(0, x1 - pad_w)
                y1_pad = max(0, y1 - pad_h)
                x2_pad = min(w_img, x2 + pad_w)
                y2_pad = min(h_img, y2 + pad_h)
                crop = img_raw[y1_pad:y2_pad, x1_pad:x2_pad]

                if crop.size > 0:
                    h_p, w_p = crop.shape[:2]
                    p_det_in = cv2.resize(crop, (int(np.ceil(w_p / 32) * 32), int(np.ceil(h_p / 32) * 32)))
                    blob_det = p_det_in.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0
                    det_outs = session_det.run(None, {session_det.get_inputs()[0].name: blob_det})[0]
                    text_boxes = get_boxes_from_map(det_outs)
                    text_boxes = sorted(text_boxes, key=lambda b: (box_to_center(b)[1] // 10, box_to_center(b)[0]))

                    temp_text = ""
                    for t_box in text_boxes:
                        pts = t_box.astype(np.float32)
                        pts[:, 0] *= (w_p / p_det_in.shape[1])
                        pts[:, 1] *= (h_p / p_det_in.shape[0])

                        bx, by, bw, bh = cv2.boundingRect(pts)
                        char_crop = crop[max(0, by):min(h_p, by + bh), max(0, bx):min(w_p, bx + bw)]
                        if char_crop.size == 0: continue

                        rec_in = preprocess_rec(char_crop)
                        if rec_in is not None:
                            preds_rec = session_rec.run(None, {session_rec.get_inputs()[0].name: rec_in})[0]
                            text = ctc_decode(preds_rec, CHARS)
                            temp_text += text
                    new_text = clean_text(temp_text)
                    if len(new_text) > 5:
                        results_cache[track_id] = {"text": new_text, "score": conf_now}
                        full_plate_text = new_text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if full_plate_text:
                cv2.putText(frame, full_plate_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        out.write(frame)
        count += 1
    cap.release()
    out.release()
    return output_path