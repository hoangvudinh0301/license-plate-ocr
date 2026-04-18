import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import imageio
from preprocessing import get_plate_corners, perspective_transform, preprocess_plate
from postprocessing import process_ocr_results, extract_with_score, stable_results

model = YOLO("model/best_pl_detection.pt")
ocr = PaddleOCR(use_textline_orientation=True, lang='en')

cap = cv2.VideoCapture("samples/IMG_6917.MOV")
frames = []
results = None
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if count % 3 == 0 or results is None:
        results = model.track(frame, conf=0.6, persist=True, verbose=False)
    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None]*len(boxes)
        for box, track_id in zip(boxes, ids):
            if track_id is None:
                continue
            x1, y1, x2, y2 = map(int, box)
            crop = frame[max(0,y1-10):y2+10, max(0,x1-10):x2+10]
            plate_text = ""
            if crop.size > 0:
                corners = get_plate_corners(crop)
                warped = perspective_transform(crop, corners)
                if warped.shape[0] < 20 or warped.shape[1] < 60:
                    continue
                th, inv = preprocess_plate(warped)
                res = ocr.predict(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))
                text, score = extract_with_score(res)

                if score < 0.6:
                    res = ocr.ocr(cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR))
                plate_text = process_ocr_results(res, track_id)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            if plate_text:
                cv2.putText(frame, plate_text, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),2)
            print(plate_text)
    frame_display = cv2.resize(frame, (640,360))
    if len(frames) < 300:
        frames.append(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB))
    count += 1
imageio.mimsave("plate_recognition.gif", frames, fps=15)
cap.release()
cv2.destroyAllWindows()