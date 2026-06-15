# license-plate-ocr
## Introdution

This project implements a real-time multi-national license plate recognition system using Deep Learning and Computer Vision techniques.

This system is capable of:

* Detecting license plates from live camera streams.
* Correcting perspective distortion using keypoint-based transformation.
* Classifying license plate nationality.
* Recognizing license plate characters.
* Supporting license plates from: Vietnam, China, Laos

The project combines multiple deep learnign models including YOLO11-Pose, EfficientNet-B3, PaddleOCR and ONNX Runtime to achieve high accuracy and fast inference speed.

## System Architecture

<img width="363" height="971" alt="image" src="https://github.com/user-attachments/assets/bee70c50-0541-4bdd-a881-c24eed42150a" />

## Technologies Used

* Python: Main programming language
* OpenCV: Image processing and camera handling
* Numpy: Numerical computation
* YOLO11-Pose: License plate detection and keypoint estimation
* EfficientNet-B3: Country classification
* PaddleOCR: Character recognition
* ONNX Runtime: High-performance inference
* Roboflow: Dataset annotation
* Kaggle: Model training
* Hugging Face: Model testing and deployment

## Experimental Results

**YOLO11-Pose**

* Metric: 99.5%
* mAP50: 87.6%
* Pose mAP50: 99.5%
* Pose mAP50-95: 99.5%

**EfficientNet-B3**

* Validation Accuracy: 98.9%

**Lao OCR Fine-Tuning**

* Accuracy: 81.97%
* Normalized Edit Distance: 95.17%

## Project Structure

```text
lincense-plate-ocr/
│
├── Documents/
│ 
├── SourceCode/
│   │
│   ├── .gradio/
│   │
│   ├── dict/
│   │   ├── china_dict.txt
│   │   ├── en_dict.txt
│   │   └── laos_dict_fn.txt
│   │
│   ├── fonts/
│   │   ├── NotoSans_Condensed-Bold.ttf
│   │   ├── NotoSansLao_Condensed-Bold.ttf
│   │   └── NotoSansSC-Bold.ttf
│   │
│   ├── function/
│   │   └── onnx_utils.py
│   │
│   ├── model/
│   │   ├── best_pl_detection.pt
│   │   ├── best_pl_detecton.onnx
│   │   ├── china_rec.onnx
│   │   ├── lao_rec.onnx
│   │   ├── plate_classification.onnx
│   │   ├── plate_classification.onnx.data
│   │   ├── plate_keypoint_detection.onnx
│   │   ├── plate_keypoint_detection.pt
│   │   ├── ppocr_det_sim.onnx
│   │   └── ppocr_rec_sim.onnx
│   │
│   ├── results/
│   │
│   ├── training/
│   │   ├── crnn-read-plate.ipynb
│   │   ├── fine-tune-efficientnet-b3.ipynb
│   │   ├── fine-tune-paddleocr-for-lao-plate.ipynb
│   │   ├── finetune_yolo_keypoints.ipynb
│   │   ├── paddleOCR.ipynb
│   │   └── pt_detection.ipynb
│   │
│   ├── uploads/
│   │
│   ├── app.py
│   ├── cam.py
│   ├── img.py
│   ├── lao_plate_ocr.py
│   ├── video.py
│   └── requirements.txt
│
├── .gitignore
└── README.md
```
## Installation

### 1. Clone repository

```text

```
### 2. Create the Python Environment

```text
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
## 3. Run

Start webcam recognition:
```text
python cam.py
```
Controls:

* S to start recognition
* P to pause recognition
* ESC to exit

## Inference result

<img width="605" height="314" alt="image" src="https://github.com/user-attachments/assets/ce430495-a06e-4cb2-89dc-7370531c560c" />

## Future Work

* Support additional countries.
* Improve OCR accuracy on low-quality images.
* Deploy on edge devices.
* Develop web and mobile applications.
* Integrate vehicle tracking and management systems.

## Author

Dinh Hoang Vu



