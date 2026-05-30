import os
import numpy as np
import onnxruntime as ort
import cv2
from function.onnx_utils import (load_chars, letterbox, warp_transform,
                                 detect_plates, preprocess_rec, ctc_decode, clean_text)

YOLO_MODEL = "model/plate_keypoint_detection.onnx"
PP_REC_MODEL = "model/ppocr_rec_sim.onnx"
DICT_PATH = "dict/en_dict.txt"

providers = ["CPUExecutionProvider"]
session_yolo = ort.InferenceSession(YOLO_MODEL, providers=providers)
session_rec = ort.InferenceSession(PP_REC_MODEL, providers=providers)
CHARS = load_chars(DICT_PATH)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

running = False
frame_count = 0
last_plate_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break
    display = frame.copy()
    cv2.putText(display,"Press S to Start | P to Pause | ESC to Exit",(20, 30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 255),2 )
    if running:
        frame_count += 1
        img_lb, ratio, pad = letterbox(frame)
        blob = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0).astype(np.float32) / 255.0
        outputs = session_yolo.run(None, {session_yolo.get_inputs()[0].name: blob})
        results = detect_plates(outputs, ratio, pad)
        plate_texts = []
        for det in results:
            x, y, w, h = det["box"]
            x1, y1 = int(x), int(y)
            pts = np.array([[kx, ky] for kx, ky, kc in det["kpts"]], dtype=np.float32)
            warped = warp_transform(frame, pts)
            h_p, w_p = warped.shape[:2]
            plate_preview = cv2.resize(warped, (w_p*2, h_p*2))
            px = 10
            py = 80
            display[
                py:py + h_p*2,
                px:px + w_p*2
            ] = plate_preview
            if h_p > w_p * 0.45:
                upper = warped[:h_p//2]
                lower = warped[h_p//2:]
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
            cv2.putText(display, f"Plate: {plate_text}", (20, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            plate_texts.append(plate_text)
    cv2.imshow("Realtime Lincense Plate Recognition", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"): running = True
    elif key == ord("p"): running = False
    elif key == 27: break
cap.release()
cv2.destroyAllWindows()