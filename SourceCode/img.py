import os
import cv2
import numpy as np
import onnxruntime as ort
from function.onnx_utils import load_chars, letterbox, detect_plates, warp_transform, preprocess_rec, ctc_decode, clean_text

YOLO_MODEL = "model/plate_keypoint_detection.onnx"
PP_REC_MODEL = "model/ppocr_rec_sim.onnx"
DICT_PATH = "en_dict.txt"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)
providers = ["CPUExecutionProvider"]

session_yolo = ort.InferenceSession(YOLO_MODEL, providers=providers)
session_rec = ort.InferenceSession(PP_REC_MODEL, providers=providers)
CHARS = load_chars(DICT_PATH)

def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img = image.copy()
    img_lb, ratio, pad = letterbox(image)
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
        warped = warp_transform(image, pts)
        h_p, w_p = warped.shape[:2]
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
        cv2.polylines(img, [pts.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(img, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
        plate_texts.append(plate_text)

    output_path = os.path.join(OUTPUT_DIR, "pl_recognition_1.png")
    cv2.imwrite(output_path, img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, "\n".join(plate_texts)