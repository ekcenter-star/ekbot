"""
scanner.py
CamScanner-style document cleaning pipeline.

Steps:
  1. Find document edges → perspective-warp flat (deskew + crop to paper only)
  2. Background division  → remove shadows/yellow tint
  3. Histogram stretch    → push paper brightness to pure white (255)
  4. CLAHE               → crisp ink without blowing highlights
  5. Sharpen             → sharp text edges
  6. Size cap            → Telegram-friendly output
"""

import cv2
import numpy as np


MAX_OUTPUT_DIM = 2480   # ~A4 at 200 dpi — big enough for Cliniko, small enough for Telegram
BG_BLUR_SIGMA  = 60     # larger = smoother background estimate


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s    = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect


def _is_valid_quad(pts: np.ndarray, img_area: int) -> bool:
    """
    Reject quads that are clearly wrong:
    - Too small (< 15 % of frame)
    - Wildly non-rectangular (one angle < 45°)
    """
    area = cv2.contourArea(pts.reshape(4, 1, 2))
    if area < 0.15 * img_area:
        return False
    # Check all 4 interior angles are between 45° and 135°
    pts = pts.reshape(4, 2)
    for i in range(4):
        p0 = pts[(i - 1) % 4]
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        v1 = p0 - p1
        v2 = p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
        if angle < 45 or angle > 135:
            return False
    return True


def _find_document_contour(image: np.ndarray):
    """
    Robust document corner detection — tries multiple Canny thresholds and
    polygon approximation levels so it works even with messy backgrounds
    (tables, hands, legs in frame).
    Returns a 4-point contour in original image coordinates, or None.
    """
    h, w  = image.shape[:2]
    scale = 800.0 / max(h, w)
    small = cv2.resize(image, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 75, 75)

    img_area = small.shape[0] * small.shape[1]

    # Try progressively looser edge detection settings
    canny_params   = [(30, 120), (20, 80), (50, 160), (10, 50)]
    approx_epsilons = [0.02, 0.03, 0.05]

    for lo, hi in canny_params:
        edged = cv2.Canny(gray, lo, hi)
        edged = cv2.dilate(edged, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours     = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        for c in contours:
            peri = cv2.arcLength(c, True)
            for eps in approx_epsilons:
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2)
                    if _is_valid_quad(pts, img_area):
                        # Scale back to original image coordinates
                        return (pts / scale).astype("float32")

    return None  # Could not detect document edges


def _find_document_by_brightness(image: np.ndarray):
    """
    Fallback detector: white paper is the BRIGHTEST large region in the photo.
    Uses Otsu threshold to isolate bright areas, then finds the largest one.
    Works even when the paper has no clear edge contrast against background.
    Returns 4 corner points or None.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu automatically picks the best threshold to separate paper from background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up noise: close small holes, remove small blobs
    kernel = np.ones((15, 15), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Take the largest bright blob — that should be the paper
    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)
    img_area = h * w

    if area < 0.15 * img_area:
        return None  # too small — probably not the document

    # Get tight bounding rectangle
    rx, ry, rw, rh = cv2.boundingRect(largest)

    # If the bounding box covers almost the entire image (> 85%), it means the
    # background is the same color as the paper (e.g. white marble table) and we
    # failed to isolate the letter. We should reject this so the user gets the
    # voice feedback to retake the photo.
    if (rw * rh) > 0.85 * img_area:
        return None

    pad = 10
    rx  = max(0,     rx  - pad)
    ry  = max(0,     ry  - pad)
    rw  = min(w - rx, rw + pad * 2)
    rh  = min(h - ry, rh + pad * 2)

    return np.array(
        [[rx, ry], [rx + rw, ry], [rx + rw, ry + rh], [rx, ry + rh]],
        dtype="float32",
    )


def _warp(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect   = _order_points(pts)
    tl, tr, br, bl = rect

    w = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    h = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M   = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w, h))


# ---------------------------------------------------------------------------
# Enhancement
# ---------------------------------------------------------------------------

def _background_divide(image: np.ndarray) -> np.ndarray:
    """
    Flat-field correction: channel / blur(channel) * 255
    Removes yellow tint, uneven lighting, and shadows entirely.
    """
    img_f  = image.astype(np.float32)
    result = np.empty_like(img_f)
    for ch in range(3):
        channel    = img_f[:, :, ch]
        background = cv2.GaussianBlur(channel, (0, 0), sigmaX=BG_BLUR_SIGMA)
        result[:, :, ch] = np.clip(channel / (background + 1.0) * 255.0, 0, 255)
    return result.astype(np.uint8)


def _stretch_to_white(image: np.ndarray) -> np.ndarray:
    """
    Histogram stretch: make the brightest 3 % of pixels pure white (255),
    darken the darkest 1 % to pure black. This pushes paper to white and
    ink to black without losing colour in stamps/logos.
    """
    gray         = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    p_dark, p_light = np.percentile(gray, [1, 97])

    if p_light <= p_dark:
        return image  # degenerate image, skip

    scale = 255.0 / (p_light - p_dark)
    result = np.empty_like(image, dtype=np.float32)
    for ch in range(3):
        ch_f = image[:, :, ch].astype(np.float32)
        result[:, :, ch] = np.clip((ch_f - p_dark) * scale, 0, 255)
    return result.astype(np.uint8)


def _boost_contrast(image: np.ndarray) -> np.ndarray:
    """CLAHE on L channel — local contrast without blowing highlights."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _sharpen(image: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=2)
    return cv2.addWeighted(image, 1.8, blur, -0.8, 0)


def _resize_cap(image: np.ndarray) -> np.ndarray:
    h, w    = image.shape[:2]
    longest = max(h, w)
    if longest <= MAX_OUTPUT_DIM:
        return image
    scale = MAX_OUTPUT_DIM / longest
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_document(image_bytes: bytes) -> tuple[bytes, bool]:
    """
    Takes raw photo bytes → returns (cleaned_PNG_bytes, letter_detected).

    letter_detected = True  → letter was found and cropped cleanly
    letter_detected = False → letter not found (too far, out of frame)
                              caller should send a feedback voice message

    Crop strategy (tries each in order):
      1. Edge-based contour detection — best for angled/tilted papers
      2. Brightness-based segmentation — fallback for flat photos with background
      3. Full frame — last resort (letter_detected = False)
    """
    arr   = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")

    letter_detected = False

    # 1. Edge-based detection — only this counts as "letter found"
    #    Requires clear 4-corner rectangle + perspective warp.
    #    If this fails it means the photo is too far, too messy, or out of frame.
    contour = _find_document_contour(image)
    if contour is not None:
        working = _warp(image, contour)
        letter_detected = True
    else:
        # 2. Brightness fallback — if we find a bright box, we consider it a successful crop
        bright_box = _find_document_by_brightness(image)
        if bright_box is not None:
            pts = _order_points(bright_box)
            tl, tr, br, bl = pts
            x1 = int(min(tl[0], bl[0]))
            y1 = int(min(tl[1], tr[1]))
            x2 = int(max(tr[0], br[0]))
            y2 = int(max(bl[1], br[1]))
            working = image[y1:y2, x1:x2]
            letter_detected = True
        else:
            # 3. Last resort — process full frame, flag as not detected
            working = image
            letter_detected = False

    working = _background_divide(working)
    working = _stretch_to_white(working)
    working = _boost_contrast(working)
    working = _sharpen(working)
    working = _resize_cap(working)

    ok, buf = cv2.imencode(".png", working)
    if not ok:
        raise ValueError("Could not encode output image")
    return buf.tobytes(), letter_detected
