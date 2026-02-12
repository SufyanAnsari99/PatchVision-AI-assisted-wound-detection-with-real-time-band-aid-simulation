
import cv2
import numpy as np
import glob

def detect_wound(image):
    """Detect reddish wound area in the arm image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red color range (two ranges because red wraps in HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 + mask2

    # Morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Assume largest red area is wound
    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < 500:
        return None

    rect = cv2.minAreaRect(largest_contour)
    return rect  # ((x, y), (w, h), angle)


def overlay_bandaid(image, bandaid, rect):
    """Overlay rotated and scaled band-aid onto wound."""
    (center_x, center_y), (w, h), angle = rect

    # Resize band-aid slightly larger than wound
    scale_w = int(w * 1.5)
    scale_h = int(h * 1.5)

    bandaid_resized = cv2.resize(bandaid, (scale_w, scale_h))

    # Rotate band-aid
    M = cv2.getRotationMatrix2D((scale_w//2, scale_h//2), angle, 1)
    bandaid_rotated = cv2.warpAffine(
        bandaid_resized,
        M,
        (scale_w, scale_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,0,0,0)
    )

    # Coordinates for placement
    x1 = int(center_x - scale_w // 2)
    y1 = int(center_y - scale_h // 2)
    x2 = x1 + scale_w
    y2 = y1 + scale_h

    # Check boundaries
    if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        return image

    overlay = image.copy()

    # Separate alpha channel
    alpha = bandaid_rotated[:, :, 3] / 255.0
    for c in range(3):
        overlay[y1:y2, x1:x2, c] = (
            alpha * bandaid_rotated[:, :, c] +
            (1 - alpha) * overlay[y1:y2, x1:x2, c]
        )

    return overlay


def process_image(image_path, bandaid):
    image = cv2.imread(image_path)
    rect = detect_wound(image)

    if rect is None:
        print(f"No wound detected in {image_path}")
        return

    result = overlay_bandaid(image, bandaid, rect)

    # Show before & after
    combined = np.hstack((image, result))
    output_name = "result_" + image_path
    cv2.imwrite(output_name, combined)
    print(f"Saved result as {output_name}")



if __name__ == "__main__":
    # Load transparent band-aid PNG
    bandaid = cv2.imread("bandaid.png", cv2.IMREAD_UNCHANGED)

    # Process at least 3 images
    images = glob.glob("arm*.jpg")

    for img_path in images:
        print(f"Processing {img_path}...")
        process_image(img_path, bandaid)

