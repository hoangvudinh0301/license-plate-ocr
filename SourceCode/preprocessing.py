import cv2
import numpy as np

def resize(img, scale):
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def enhance_contrast(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def denoise(img):
    return cv2.bilateralFilter(img, 9, 75, 75)

def threshold_adaptive(img):
    return cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

def invert(img):
    return cv2.bitwise_not(img)

def preprocess_plate(img):
    gray = to_gray(img)
    gray = enhance_contrast(gray)
    gray = denoise(gray)
    th = threshold_adaptive(gray)
    inv = invert(th)
    return th, inv

def order_point(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(image, pts):
    rect = order_point(pts)
    (tl, tr, br, bl) = rect
    w = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    h = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w, h))

def get_plate_corners(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            (x, y, w, h) = cv2.boundingRect(pts)
            ratio = w/float(h)

            if 2 < ratio < 6:
                return pts

    h, w = crop.shape[:2]
    return np.array([[0, 0], [w, 0], [w, h], [0, h]])




