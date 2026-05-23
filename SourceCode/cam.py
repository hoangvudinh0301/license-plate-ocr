import os
import numpy as np
import onnxruntime as ort
import cv2
from function.onnx_utils import (load_chars, letterbox, warp_transform,
                                 detect_plates, preprocess_rec, ctc_decode, clean_text)
import supervision as sv

YOLO_MODEL = "model/plate_keypoint_detection.onnx"
PP_REC_MODEL = "model/ppocr_rec_sim.onnx"
DICT_PATH = "en_dict.txt"
VIDEO_PATH = "samples/IMG_6917.MOV"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

providers = ["CPUExecutionProvider"]
session_yolo = ort.InferenceSession(YOLO_MODEL, providers=providers)
session_rec = ort.InferenceSession(PP_REC_MODEL, providers=providers)
CHARS = load_chars(DICT_PATH)

def process_video(video_path):
    tracker = sv.ByteTrack()
    results_cache = {}
    results = []

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_path = "results/output_result.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    detections = sv.Detections.empty()
    count = 0
    DETECT_W = 640
    scale = DETECT_W / width if width > DETECT_W else 1.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % 5 == 0:
            if scale < 1.0:
                small = cv2.resize(frame, (int(width * scale), int(height * scale)))
            else:
                small = frame
            img_lb, ratio, pad = letterbox(small)
            blob = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
            blob = blob.transpose(2, 0, 1)
            blob = np.expand_dims(blob, axis=0).astype(np.float32) / 255.0
            outputs = session_yolo.run(None, {session_yolo.get_inputs()[0].name: blob})
            results = detect_plates(outputs, ratio, pad)

            if scale < 1.0 and results:
                inv = 1.0 / scale
                for r in results:
                    r["box"] = [v * inv for v in r["box"]]
                    r["kpts"] = [(kx * inv, ky * inv, kc) for kx, ky, kc in r["kpts"]]

            if results is not None and len(results) > 0:
                detections = sv.Detections(
                    xyxy=np.array([
                        [
                            b["box"][0],
                            b["box"][1],
                            b["box"][0] + b["box"][2],
                            b["box"][1] + b["box"][3]
                        ]
                        for b in results
                    ], dtype=np.float32),
                    confidence=np.array([b["score"] for b in results], dtype=np.float32),
                    class_id=np.zeros(len(results), dtype=int)
                )
                detections = tracker.update_with_detections(detections)
            else: detections = sv.Detections.empty()

        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            conf_now = detections.confidence[i]
            track_id = detections.tracker_id[i] if detections.tracker_id is not None else None
            if track_id is None:
                continue

            cached_data = results_cache.get(track_id, {"text": "", "score": 0.0})
            full_plate_text = cached_data["text"]
            if not full_plate_text or conf_now > cached_data["score"]:
                kpts = results[i]["kpts"]
                pts = np.array([[kx, ky] for kx, ky, kc in kpts], dtype=np.float32)
                warped = warp_transform(frame, pts)
                h_p, w_p = warped.shape[:2]
                if h_p > w_p * 0.45:
                    upper = warped[:h_p // 2]
                    lower = warped[h_p // 2:]
                    plate_text = ""
                    for crop in [upper, lower]:
                        rec_in = preprocess_rec(crop)
                        if rec_in is None:
                            continue
                        preds = session_rec.run(None, {session_rec.get_inputs()[0].name: rec_in})[0]
                        plate_text += ctc_decode(preds, CHARS)
                else:
                    rec_in = preprocess_rec(warped)
                    if rec_in is None:
                        continue
                    preds = session_rec.run(None, {session_rec.get_inputs()[0].name: rec_in})[0]
                    plate_text = ctc_decode(preds, CHARS)
                plate_text = clean_text(plate_text)
                if len(plate_text) > 5:
                    results_cache[track_id] = {"text": plate_text, "score": conf_now}
                    full_plate_text = plate_text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if full_plate_text:
                cv2.putText(frame, full_plate_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        out.write(frame)
        count += 1
    cap.release()
    out.release()
    return output_path