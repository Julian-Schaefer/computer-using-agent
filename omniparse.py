from typing import Optional
from PIL import Image
from utils import (
    get_yolo_model,
    check_ocr_box,
    get_som_labeled_img,
    caption_model_processor,
)
import io
import base64
import os
import time


yolo_model = get_yolo_model(model_path="weights/icon_detect/model.pt")


def process_image(
    image_input, box_threshold=0.3, iou_threshold=0.5, use_paddleocr=True, imgsz=3200
):
    """
    Process an image and return the parsed content and processed image.

    Args:
        image_input: PIL Image object or path to image file
        box_threshold: Threshold for bounding box detection (default: 0.3)
        iou_threshold: IOU threshold for NMS (default: 0.5)
        use_paddleocr: Whether to use PaddleOCR for text recognition (default: True)
        imgsz: Image size for processing (default: 3200)

    Returns:
        dict: Dictionary containing 'parsed_content' and base64-encoded 'image'

    Raises:
        Exception: If image processing fails
    """
    try:
        # Handle string input (file path)
        if isinstance(image_input, str):
            image_input = Image.open(image_input)

        # Process the image
        processed_image, parsed_content = process(
            image_input=image_input,
            box_threshold=box_threshold,
            iou_threshold=iou_threshold,
            use_paddleocr=use_paddleocr,
            imgsz=imgsz,
        )

        # Save processed image to bytes
        img_byte_arr = io.BytesIO()
        processed_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8"), parsed_content

    except Exception as e:
        # Re-raise the exception to be handled by the caller
        raise Exception(f"Error processing image: {str(e)}")


def process(
    image_input, box_threshold, iou_threshold, use_paddleocr, imgsz
) -> Optional[Image.Image]:
    if not os.path.exists("imgs"):
        os.makedirs("imgs")

    image_save_path = "imgs/saved_image_demo_" + str(time.time()) + ".png"
    image_input.save(image_save_path)
    image = Image.open(image_save_path)

    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        "text_scale": 0.8 * box_overlay_ratio,
        "text_thickness": max(int(2 * box_overlay_ratio), 1),
        "text_padding": max(int(3 * box_overlay_ratio), 1),
        "thickness": max(int(3 * box_overlay_ratio), 1),
    }
    # import pdb; pdb.set_trace()

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_save_path,
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": 0.9},
        use_paddleocr=use_paddleocr,
    )
    text, ocr_bbox = ocr_bbox_rslt
    # print('prompt:', prompt)
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_save_path,
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
    )
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    print("finish processing")
    parsed_content_list = "\n".join(
        [f"icon {i}: " + str(v) for i, v in enumerate(parsed_content_list)]
    )
    # parsed_content_list = str(parsed_content_list)
    return image, str(parsed_content_list)
