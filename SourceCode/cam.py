import os
import numpy as np
import onnxruntime as ort
import cv2
from function.onnx_utils import (load_chars, letterbox, warp_transform,
                                 detect_plates, preprocess_rec, ctc_decode, clean_text, plate_classification, put_unicode_text)
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

YOLO_MODEL = "model/plate_keypoint_detection.onnx"
CLASSIFY_MODEL = "model/plate_classification.onnx"
VN_REC_MODEL = "model/ppocr_rec_sim.onnx"
CH_REC_MODEL = "model/china_rec.onnx"
LA_REC_MODEL = "model/lao_rec.onnx"
VN_DICT_PATH = "dict/en_dict.txt"
CH_DICT_PATH = "dict/china_dict.txt"
LA_DICT_PATH = "dict/laos_dict_fn.txt"

FONT_MAPPING = {
    "VietNam": "fonts/NotoSans_Condensed-Bold.ttf",
    "Lao": "fonts/NotoSansLao_Condensed-Bold.ttf",
    "China": "fonts/NotoSansSC-Bold.ttf"
}

OCR_CONFIG = {
    "VietNam": {"model": VN_REC_MODEL, "dict": VN_DICT_PATH},
    "China": {"model": CH_REC_MODEL, "dict": CH_DICT_PATH},
    "Lao": {"model": LA_REC_MODEL, "dict": LA_DICT_PATH}
}

providers = ["CPUExecutionProvider"]
session_yolo = ort.InferenceSession(YOLO_MODEL, providers=providers)
sessions = {
    nation: ort.InferenceSession(cfg["model"], providers=providers)
    for nation, cfg in OCR_CONFIG.items()
}
session_cls = ort.InferenceSession(CLASSIFY_MODEL, providers=providers)

DICT_DATA = {
    nation: load_chars(cfg["dict"])
    for nation, cfg in OCR_CONFIG.items()
}
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 680)

running = False
frame_count = 0

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
            warped_pil = Image.fromarray(warped)
            nation = plate_classification(warped_pil, session_cls)
            session_rec = sessions[nation]
            input_shape = session_rec.get_inputs()[0].shape
            model_height = input_shape[2] if isinstance(input_shape[2], int) else 48
            model_width = input_shape[3] if isinstance(input_shape[3], int) else 320
            CHARS = load_chars(OCR_CONFIG[nation]["dict"])
            h_p, w_p = warped.shape[:2]
            plate_preview = cv2.resize(warped, (w_p*2, h_p*2))
            px = 10
            py = 100
            display[py:py + h_p*2, px:px + w_p*2] = plate_preview
            if h_p > w_p * 0.45:
                upper = warped[:h_p//2]
                lower = warped[h_p//2:]
                plate_text = ""
                for crop in [upper, lower]:
                    rec_in = preprocess_rec(crop, target_height=model_height, target_width=model_width)
                    if rec_in is None:
                        continue
                    preds = session_rec.run(None, {session_rec.get_inputs()[0].name: rec_in})[0]
                    plate_text += ctc_decode(preds, CHARS)
            else:
                rec_in = preprocess_rec(warped, target_height=model_height, target_width=model_width)
                if rec_in is None:
                    continue
                preds = session_rec.run(None, {session_rec.get_inputs()[0].name: rec_in})[0]
                plate_text = ctc_decode(preds, CHARS)

            plate_text = clean_text(plate_text)
            active_font = FONT_MAPPING.get(nation, "fonts/NotoSans_Condensed-Bold.ttf")
            font = FontProperties(fname=active_font, size=16)
            plt.imshow(cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
            plt.title(f"Nation: {nation} | Plate: {plate_text}", fontproperties=font)
            plt.axis("off")
            plt.show()


    cv2.imshow("Realtime Lincense Plate Recognition", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"): running = True
    elif key == ord("p"): running = False
    elif key == 27: break
cap.release()
cv2.destroyAllWindows()