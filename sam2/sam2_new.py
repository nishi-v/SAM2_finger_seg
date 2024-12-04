import os
from pathlib import Path
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import logging
import time
from PIL import Image

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
dir = Path(os.getcwd())
data_dir = dir/'test_data'
images_dir = data_dir/'images'  
bbox_dir = data_dir/'bbox'  
output_dir = data_dir/'new/sam2.1_t' 
bbox_output_dir = output_dir/'bbox'
segments_dir = output_dir/'segments'
mask_dir = output_dir/'masks' 
weights_dir = dir/'weights_new'
logs_dir = output_dir/'logs'

# Finger class mapping and colors
FINGER_CLASS_MAP = {
    0: 'Index',
    1: 'Middle', 
    2: 'Pinky',
    3: 'Ring'
}

COLORS = {
    0: (255, 0, 0),    # Blue for Index
    1: (0, 255, 0),    # Green for Middle
    2: (0, 0, 255),    # Red for Pinky
    3: (255, 255, 0),  # Cyan for Ring
}

# Create output directories
for directory in [output_dir, bbox_output_dir, segments_dir, mask_dir, logs_dir]:
    directory.mkdir(parents=True, exist_ok=True)

# Log file setup
log_file = logs_dir / 'sam2.1_t_mask_generation.log'
logging.basicConfig(filename=str(log_file), level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')  # Overwrite log file each run

# Load SAM2 model
checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

try:
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
except Exception as e:
    logging.error(f"Error loading SAM2 model: {e}")
    raise

def load_bboxes(bbox_file: str):
    """
    Load bounding boxes from a text file.
    Expected format: class_index x_min y_min x_max y_max
    """
    bboxes = []
    with open(bbox_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_index = int(parts[0])
            x_min = float(parts[1])
            y_min = float(parts[2])
            x_max = float(parts[3])
            y_max = float(parts[4])
            bboxes.append([class_index, x_min, y_min, x_max, y_max])
    return bboxes

def normalize_bbox_to_pixels(bbox, image_width, image_height):
    """
    Convert normalized bbox coordinates to pixel coordinates.
    """
    class_index, x_min, y_min, x_max, y_max = bbox
    x_min_pixel = int(x_min * image_width)
    y_min_pixel = int(y_min * image_height)
    x_max_pixel = int(x_max * image_width)
    y_max_pixel = int(y_max * image_height)
    return [class_index, x_min_pixel, y_min_pixel, x_max_pixel, y_max_pixel]

def smooth_mask(mask, kernel_size=5):
    """
    Apply Gaussian blur to smooth the mask
    """
    return cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)

def generate_masks_for_image(image_path: Path, predictor: SAM2ImagePredictor, pixel_bboxes: list[list[int]]) -> tuple:
    """
    Generate masks for an image using SAM2 predictor.
    """
    try:
        # Convert bounding boxes to input format
        input_boxes = np.array([bbox[1:] for bbox in pixel_bboxes])

        # Load the image
        image = cv2.imread(str(image_path))
        image_pil = Image.open(image_path)

        # Set the image for prediction
        predictor.set_image(image_pil)

        # Measure prediction time
        start_time = time.time()
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        prediction_time = time.time() - start_time

        logging.info(f"Processed {image_path.name} in {prediction_time:.2f} seconds")
        return masks, scores, input_boxes, image, prediction_time

    except Exception as e:
        logging.error(f"Error generating masks for {image_path.name}: {e}")
        return None, None, None, None, None

def process_images():
    """
    Process all images in the images directory and log the average prediction time.
    """
    prediction_times = []

    for image_file in os.listdir(images_dir):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            try:
                image_path = images_dir / image_file
                bbox_file = bbox_dir / f"box_{image_file.split('.')[0]}.txt"

                if not bbox_file.exists():
                    logging.warning(f"Bounding box file not found for {image_file}")
                    continue

                # Load and normalize bounding boxes
                image_cv = cv2.imread(str(image_path))
                image_height, image_width = image_cv.shape[:2]
                
                bboxes = load_bboxes(str(bbox_file))
                pixel_bboxes = [normalize_bbox_to_pixels(bbox, image_width, image_height) for bbox in bboxes]
            
                # Generate masks and record prediction time
                masks, scores, input_boxes, original_image, prediction_time = generate_masks_for_image(image_path, predictor, pixel_bboxes)

                if prediction_time:
                    prediction_times.append(prediction_time)
                
                # Create a blank mask overlay for all fingers
                mask_overlay = np.zeros_like(original_image, dtype=np.uint8)

                # Process each finger segmentation
                for (finger_bbox, mask) in zip(pixel_bboxes, masks):
                    class_index, x_min, y_min, x_max, y_max = finger_bbox
                    color = COLORS[class_index]

                    img_new = original_image.copy()
                
                    # Draw bounding box on the image
                    cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), color, 2)

                    # Convert mask to binary and prepare for polygon drawing
                    binary_mask = (mask[0] > 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Fill the mask for this finger with its corresponding color
                    if contours:
                        cv2.drawContours(mask_overlay, contours, -1, color, -1)
                        
                        logging.info(f"Processed {FINGER_CLASS_MAP[class_index]} finger")

                # Smooth the mask overlay
                smoothed_mask = smooth_mask(mask_overlay)

                # Save the image with bounding boxes
                bbox_image_path = bbox_output_dir/f"bbox_{image_file}"
                cv2.imwrite(str(bbox_image_path), original_image)
                logging.info(f"Saved image with bounding boxes: {bbox_image_path}")

                # Blend the smoothed mask overlay with the original image
                alpha = 0.5  # Transparency level
                masked_image = cv2.addWeighted(img_new, 1 - alpha, smoothed_mask.astype(np.uint8), alpha, 0)

                # Save the final image with all finger masks
                masked_image_path = mask_dir/f"masked_{image_file}"
                cv2.imwrite(str(masked_image_path), masked_image)
                logging.info(f"Saved image with all finger masks: {masked_image_path}")

                # Save the pure smoothed mask overlay for debugging
                mask_path = segments_dir/f"mask_{image_file}"
                cv2.imwrite(str(mask_path), smoothed_mask.astype(np.uint8))
                logging.info(f"Saved smoothed mask overlay: {mask_path}")

            except Exception as e:
                logging.error(f"Error processing {image_file}: {e}")

        logging.info("Finger Segmentation Successful!")

    if prediction_times:
        avg_time = sum(prediction_times) / len(prediction_times)
        logging.info(f"Average prediction time per image: {avg_time:.2f} seconds")
    else:
        logging.warning("No images processed.")

if __name__ == "__main__":
    process_images()