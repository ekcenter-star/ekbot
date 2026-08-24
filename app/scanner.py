"""
scanner.py
Turns a phone photo of a paper document (consult letter / invoice) into a
clean, color, "scanned-looking" image — CamScanner style.

Pipeline:
  1. Find the document's 4 corners in the photo
  2. Perspective-warp it flat (deskew)
  3. Auto white-balance so the paper looks white, not yellow/gray
  4. Boost local contrast (CLAHE) so ink is crisp without blowing out stamps/signatures
  5. Sharpen text edges

If no clean 4-corner document is found (e.g. photo is already fairly flat,
or edges are cut off), we skip the perspective warp and just run the
enhancement steps on the original frame — safer than guessing wrong and
cropping into the letter content.
"""

import cv2
import numpy as np


MAX_OUTPUT_DIM = 2000  # cap longest side so files stay Telegram-friendly


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _find_document_contour(image: np.ndarray):
    """Return a 4-point contour for the document, or None if not confident."""
    h, w = image.shape[:2]
    ratio = 1000.0 / w if w > 1000 else 1.0
    small = cv2.resize(image, (int(w * ratio), int(h * ratio)))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 60, 160)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    small_area = small.shape[0] * small.shape[1]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            # Require the document to take up a healthy chunk of the frame,
            # otherwise it's probably not the actual page edges.
            if area > 0.35 * small_area:
                return (approx.reshape(4, 2) / ratio).astype("float32")
    return None


def _warp(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_width, max_height))


def _white_balance(image: np.ndarray) -> np.ndarray:
    """Simple gray-world white balance so the paper reads as white."""
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(result)
    avg_a = np.average(a)
    avg_b = np.average(b)
    a = cv2.add(a, ((avg_a - 128) * (l / 255.0) * -1.1).astype(np.uint8))
    b = cv2.add(b, ((avg_b - 128) * (l / 255.0) * -1.1).astype(np.uint8))
    result = cv2.merge([l, a, b])
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def _boost_contrast(image: np.ndarray) -> np.ndarray:
    """CLAHE on the luminance channel — crisps up text while keeping color."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _sharpen(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def _resize_cap(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= MAX_OUTPUT_DIM:
        return image
    scale = MAX_OUTPUT_DIM / longest
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def clean_document(image_bytes: bytes) -> bytes:
    """
    Main entry point. Takes raw photo bytes, returns cleaned PNG bytes.
    Keeps color (stamps/signatures/logos stay in color) — no B&W thresholding.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")

    contour = _find_document_contour(image)
    working = _warp(image, contour) if contour is not None else image

    working = _white_balance(working)
    working = _boost_contrast(working)
    working = _sharpen(working)
    working = _resize_cap(working)

    ok, buf = cv2.imencode(".png", working)
    if not ok:
        raise ValueError("Could not encode output image")
    return buf.tobytes()
