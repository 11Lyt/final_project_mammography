import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import cv2
import pydicom

from PIL import Image, ImageDraw, ImageFont

# ============================================================
# CONFIG
# ============================================================

# Folder containing FP_gradio_app.py
APP_ROOT = Path(__file__).resolve().parent

# working_files/
PROJECT_ROOT = APP_ROOT.parent

# Folder containing YOLOv9's models/ and utils/
YOLOV9_ROOT = PROJECT_ROOT / "yolov9"

# Add YOLOv9 root to Python's import path
if str(YOLOV9_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV9_ROOT))

MODEL_WEIGHTS = str(
    PROJECT_ROOT
    / "draft_models"
    / "yolov9"
    / "runs"
    / "train"
    / "final_radimagenet_densenet121"
    / "weights"
    / "last.pt"
)

IMG_SIZE = 1024

# Detection thresholds
# NOTE: 0.046104 is the Youden's J threshold found during tuning. 
# More conventional value: 0.25 (for a cleaner demo UI)
CONF_THRESH = 0.046104
IOU_THRESH = 0.45

CLASS_NAMES = ["benign", "malignant"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Colors for drawing boxes per class (RGB)
BOX_COLORS = {
    "benign": (66, 135, 245),      # blue
    "malignant": (235, 64, 52),    # red
}

# ============================================================
# MODEL LOADING + INFERENCE
# ============================================================

_model = None


def load_model():
    """Lazy-load the YOLOv9 model once and cache it."""
    global _model

    if _model is not None:
        return _model

    print("Importing YOLO modules...")

    from models.common import DetectMultiBackend
    from utils.general import check_img_size

    print("Loading model weights...")

    model = DetectMultiBackend(
        MODEL_WEIGHTS,
        device=torch.device(DEVICE),
        fp16=False
    )

    print("Warming up model...")

    model.warmup(
        imgsz=(1, 3, IMG_SIZE, IMG_SIZE)
    )

    print("Model ready.")

    _model = model
    return _model


def load_dicom_image(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(np.float32)

    if ds.PhotometricInterpretation == "MONOCHROME1":
        img = img.max() - img

    return img


def percentile_clip(img, lower=1, upper=99):
    p_low = np.percentile(img, lower)
    p_high = np.percentile(img, upper)

    img = np.clip(img, p_low, p_high)

    return img


def normalize_0_to_1(img):
    img_min = img.min()
    img_max = img.max()

    img = (img - img_min) / (img_max - img_min + 1e-6)

    return img.astype(np.float32)


def to_three_channels(img):
    return np.stack([img, img, img], axis=-1)


def letterbox(img, new_size=1024):
    h, w = img.shape[:2]

    scale = min(new_size / w, new_size / h)

    resized_w = int(w * scale)
    resized_h = int(h * scale)

    resized_img = cv2.resize(
        img,
        (resized_w, resized_h),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = np.zeros(
        (new_size, new_size, 3),
        dtype=np.float32
    )

    pad_x = (new_size - resized_w) // 2
    pad_y = (new_size - resized_h) // 2

    canvas[
        pad_y:pad_y + resized_h,
        pad_x:pad_x + resized_w
    ] = resized_img

    return canvas


def preprocess_dicom(dicom_path):
    """Apply the same preprocessing pipeline used for model training."""

    # 1. Load raw mammogram
    img = load_dicom_image(dicom_path)

    # 2. Percentile clipping
    img = percentile_clip(img, lower=1, upper=99)

    # 3. Normalise to [0, 1]
    img = normalize_0_to_1(img)

    # 4. Convert grayscale to 3 channels
    img = to_three_channels(img)

    # 5. Letterbox to training resolution
    img = letterbox(img, new_size=IMG_SIZE)

    return img

def dicom_to_preview(dicom_path):
    """
    Convert a raw DICOM mammogram into a displayable PIL image.
    This is only for the Gradio preview.
    """

    img = load_dicom_image(dicom_path)

    img = percentile_clip(
        img,
        lower=1,
        upper=99
    )

    img = normalize_0_to_1(img)

    # Convert [0, 1] float image to [0, 255] uint8 for display
    img_uint8 = (
        img * 255
    ).clip(0, 255).astype(np.uint8)

    return Image.fromarray(img_uint8).convert("RGB")


def run_inference_single(dicom_path):
    """
    Runs the model on a single raw DICOM mammogram.

    The displayed result is the same 1024x1024 preprocessed image
    that is passed into the model, so predicted bounding boxes can
    be drawn directly without rescaling them to the original DICOM.
    """
    from utils.general import non_max_suppression

    model = load_model()

    # --------------------------------------------------------
    # 1. Apply the same preprocessing used during training
    # --------------------------------------------------------
    img = load_dicom_image(dicom_path)

    img = percentile_clip(
        img,
        lower=1,
        upper=99
    )

    img = normalize_0_to_1(img)

    img = to_three_channels(img)

    img = letterbox(
        img,
        new_size=IMG_SIZE
    )

    # --------------------------------------------------------
    # 2. Create image for displaying detections
    # --------------------------------------------------------
    display_img = (
        img * 255
    ).clip(0, 255).astype(np.uint8)

    display_img = Image.fromarray(display_img)

    # --------------------------------------------------------
    # 3. Prepare model input
    # --------------------------------------------------------
    # HWC -> CHW
    model_input = img.transpose(2, 0, 1)

    model_input = np.ascontiguousarray(model_input)

    model_input = (
        torch.from_numpy(model_input)
        .to(DEVICE)
        .float()
    )

    # Add batch dimension:
    # (3, 1024, 1024) -> (1, 3, 1024, 1024)
    if model_input.ndimension() == 3:
        model_input = model_input.unsqueeze(0)

    # --------------------------------------------------------
    # 4. Run model inference
    # --------------------------------------------------------
    with torch.no_grad():

        pred = model(model_input)

        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        pred = non_max_suppression(
            pred,
            CONF_THRESH,
            IOU_THRESH
        )

    # --------------------------------------------------------
    # 5. Extract predicted detections
    # --------------------------------------------------------
    detections = []

    for det in pred:
        if det is not None and len(det):
            for *xyxy, conf, cls in det.tolist():
                class_id = int(cls)

                if class_id < len(CLASS_NAMES):
                    cls_name = CLASS_NAMES[class_id]
                else:
                    cls_name = str(class_id)

                detections.append({
                    "box": [float(x) for x in xyxy],
                    "conf": float(conf),
                    "class": cls_name,
                })

    # --------------------------------------------------------
    # 6. Draw model-generated bounding boxes
    # --------------------------------------------------------
    annotated = draw_detections(
        display_img,
        detections
    )

    # --------------------------------------------------------
    # 7. Produce summary classification
    # --------------------------------------------------------
    label = summarize_classification(detections)

    return annotated, label


def draw_detections(pil_image: Image.Image, detections: list) -> Image.Image:
    """Draws bounding boxes + per-box labels onto a copy of the image."""
    img = pil_image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(14, img.width // 60))
    except Exception:
        font = ImageFont.load_default()

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = BOX_COLORS.get(d["class"], (52, 235, 131))
        line_w = max(2, img.width // 300)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

        tag = f"{d['class']} {d['conf']:.2f}"
        text_bbox = draw.textbbox((0, 0), tag, font=font)
        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, max(0, y1 - th - 4)), tag, fill=(255, 255, 255), font=font)

    return img


def summarize_classification(detections: list) -> str:
    """Turns the raw detections into a single bottom-line classification result."""
    if not detections:
        return "No abnormality detected"

    best = max(detections, key=lambda d: d["conf"])
    n_boxes = len(detections)
    plural = "" if n_boxes == 1 else "s"
    return f"{best['class'].capitalize()} (confidence {best['conf']:.2f}) — {n_boxes} detection{plural}"


# ============================================================
# GRADIO APP
# ============================================================

def on_upload(files):
    """Load uploaded DICOM files and show the first mammogram."""

    if not files:
        return (
            [],
            0,
            [],
            None,
            "No images uploaded yet",
            gr.update(visible=False),
            gr.update(visible=False),
            ""
        )

    dicom_paths = [f.name for f in files]

    preview = dicom_to_preview(dicom_paths[0])

    idx = 0
    counter = f"Image 1 of {len(dicom_paths)}"

    # Only show navigation buttons if there is more than 1 image
    show_navigation = len(dicom_paths) > 1

    return (
        dicom_paths,
        idx,
        [],
        preview,
        counter,
        gr.update(visible=show_navigation),
        gr.update(visible=show_navigation),
        ""
    )


def go_prev(dicom_paths, idx, results):
    if not dicom_paths:
        return None, "", idx, ""

    idx = max(0, idx - 1)

    counter = f"Image {idx + 1} of {len(dicom_paths)}"

    # Show detection result if already available,
    # otherwise show the original DICOM preview
    if results and idx < len(results):
        display_img = results[idx][0]
        label = results[idx][1]
    else:
        display_img = dicom_to_preview(dicom_paths[idx])
        label = ""

    return display_img, counter, idx, label


def go_next(dicom_paths, idx, results):
    if not dicom_paths:
        return None, "", idx, ""

    idx = min(len(dicom_paths) - 1, idx + 1)

    counter = f"Image {idx + 1} of {len(dicom_paths)}"

    if results and idx < len(results):
        display_img = results[idx][0]
        label = results[idx][1]
    else:
        display_img = dicom_to_preview(dicom_paths[idx])
        label = ""

    return display_img, counter, idx, label


def run_model(dicom_paths, idx):
    if not dicom_paths:
        return None, "No images uploaded", []

    results = [
        run_inference_single(path)
        for path in dicom_paths
    ]

    if idx >= len(results):
        idx = 0

    return (
        results[idx][0],
        results[idx][1],
        results
    )


with gr.Blocks(title="Mammography Abnormality Detection") as demo:
    gr.Markdown(
        "# Mammography Abnormality Detection\n"
        "RadImageNet-pretrained DenseNet121 + YOLOv9\n\n"
        "Upload mammogram DICOM file(s), preview them, then run detection."
    )

    dicom_paths_state = gr.State([])   # list of PIL images
    idx_state = gr.State(0)       # current index
    results_state = gr.State([])  # list of (annotated_image, label) tuples

    with gr.Row():
        file_upload = gr.File(
            label="Upload mammogram DICOM file(s)",
            file_count="multiple",
            file_types=[".dcm"],
        )

    gr.Markdown("### Loaded Mammogram")
    with gr.Row():

        prev_btn = gr.Button(
            "◀ Prev",
            scale=1,
            visible=False
        )

        image_display = gr.Image(
            label="Mammogram",
            type="pil",
            scale=6,
            height=520
        )

        next_btn = gr.Button(
            "Next ▶",
            scale=1,
            visible=False
        )

    counter_label = gr.Markdown(
        "No images uploaded yet"
    )

    run_btn = gr.Button(
        "Run Detection",
        variant="primary"
    )

    classification_label = gr.Markdown("")

    # --- wiring ---
    file_upload.change(
        fn=on_upload,
        inputs=[file_upload],
        outputs=[
            dicom_paths_state,
            idx_state,
            results_state,
            image_display,
            counter_label,
            prev_btn,
            next_btn,
            classification_label
        ]
    )

    prev_btn.click(
        fn=go_prev,
        inputs=[
            dicom_paths_state,
            idx_state,
            results_state
        ],
        outputs=[
            image_display,
            counter_label,
            idx_state,
            classification_label
        ]
    )

    next_btn.click(
        fn=go_next,
        inputs=[
            dicom_paths_state,
            idx_state,
            results_state
        ],
        outputs=[
            image_display,
            counter_label,
            idx_state,
            classification_label
        ]
    )

    run_btn.click(
        fn=run_model,
        inputs=[
            dicom_paths_state,
            idx_state
        ],
        outputs=[
            image_display,
            classification_label,
            results_state
        ]
    )


if __name__ == "__main__":
    demo.launch(share=True)
