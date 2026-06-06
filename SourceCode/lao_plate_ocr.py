import cv2
import matplotlib.pyplot as plt
import onnxruntime as ort
from function.onnx_utils import load_chars, ctc_decode, preprocess_rec

PP_REC_MODEL = "model/lao_rec.onnx"
DICT_PATH = "dict/laos_dict_fn.txt"
IMAGE_PATH = "debug.png"

providers = ["CPUExecutionProvider"]
session_rec = ort.InferenceSession(PP_REC_MODEL, providers=providers)

input_shape = session_rec.get_inputs()[0].shape
print(f"Cấu hình Shape đầu vào yêu cầu của ONNX: {input_shape}")

model_height = input_shape[2] if isinstance(input_shape[2], int) else 48
model_width = input_shape[3] if isinstance(input_shape[3], int) else 320

CHARS = load_chars(DICT_PATH)
img = cv2.imread(IMAGE_PATH)

rec_in = preprocess_rec(img, target_height=model_height, target_width=model_width)
input_name = session_rec.get_inputs()[0].name
preds = session_rec.run(None, {session_rec.get_inputs()[0].name: rec_in})[0]
text = ctc_decode(preds, CHARS)

plt.figure(figsize=(12, 6))
plt.title(f"Plate: {text}")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
plt.axis("off")
plt.show()