import re

def clean_text(text):
    if not text:
        return ""
    text = text.upper()
    text = re.sub(r'[^A-Z0-9\n]', '', text)
    return text.strip()

import numpy as np

def extract_lines(res):
    if not res or not res[0]:
        return ""
    boxes = res[0]['rec_boxes']
    texts = res[0]['rec_texts']
    items = []
    for box, text in zip(boxes, texts):
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        items.append((x_center, y_center, text))
    if not items:
        return ""
    try:
        h = res[0]['doc_preprocessor_res']['output_img'].shape[0]
        threshold = h * 0.15
    except:
        threshold = 20
    items = sorted(items, key=lambda x: x[1])
    lines = []
    current = [items[0]]
    for i in range(1, len(items)):
        mean_y = np.mean([p[1] for p in current])
        if abs(items[i][1] - mean_y) < threshold:
            current.append(items[i])
        else:
            lines.append(current)
            current = [items[i]]
    lines.append(current)
    lines = sorted(lines, key=lambda line: np.mean([p[1] for p in line]))
    result = []
    for line in lines:
        line = sorted(line, key=lambda x: x[0])
        result.append("".join(clean_text(t[2]) for t in line))
    return "-".join(result)

stable_results = {}
def extract_with_score(res):
    if res and res[0]:
        texts = res[0].get('rec_texts', [])
        scores = res[0].get('rec_scores', [])

        if not texts or not scores:
            return "", 0
        texts = [t for t in texts if len(t) >= 2]
        if not texts:
            return "", 0
        text = "".join(texts)
        score = sum(scores) / len(scores)
        return text, score
    return "", 0

def get_best_result(new_text, new_score, track_id):
    if track_id not in stable_results:
        stable_results[track_id] = (new_text, new_score)
        return new_text

    old_text, old_score = stable_results[track_id]
    if new_score > old_score or len(new_text) > len(old_text):
        stable_results[track_id] = (new_text, new_score)

    return stable_results[track_id][0]

def process_ocr_results(res, track_id):

    raw_text = extract_lines(res)
    _, score = extract_with_score(res)

    if score > 0.7 and len(raw_text) >= 6:
        return get_best_result(raw_text, score, track_id)
    else:
        return stable_results.get(track_id, ("", 0))[0]