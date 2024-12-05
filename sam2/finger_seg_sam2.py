import math

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from rembg import new_session, remove
from ultralytics import YOLO
import os
from pathlib import Path
from typing import List, Tuple, Dict

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import logging
import time
from PIL import Image
import matplotlib.pyplot as plt
from sam2.modeling.backbones.hieradet import Hiera

# Device 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
dir = Path(os.getcwd())
print(dir)
data_dir = dir/'test_data'
img_dir = data_dir/'images'
# bbox_dir = data_dir/'bbox'
output_dir = data_dir/'test/sam2_t_640_160_check_1'
bbox_dir = output_dir/'bbox'
mask_dir = output_dir/'mask'
segments_dir = output_dir/'segments'
logs_dir = output_dir/'logs'
rings_dir = data_dir/'rings'

# Ensuring output directories exist
for directory in [output_dir, bbox_dir, segments_dir, mask_dir, logs_dir]:
    directory.mkdir(parents=True, exist_ok=True)

# Model paths
model_dir = dir/'models'
model_file = model_dir/'best_20epochs.pt'

checkpoint_file = "checkpoints/sam2_hiera_tiny.pt"
model_cfg = "configs/sam2/sam2_hiera_t.yaml"

model_rembg = "u2net_human_seg"

session = new_session(model_name=model_rembg)
hand_landmarker_path = model_dir/'hand_landmarker.task'
model = YOLO(model_file, task="segment")
base_options = python.BaseOptions(model_asset_path=hand_landmarker_path)

# Initializing log file
log_file = logs_dir/'logs_sam2_hiera_t.log'
logging.basicConfig(
    filename=str(log_file), 
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize Models
try:
    # Mediapipe hands
    mp_hands = mp.solutions.hands #type:ignore
    hand_model = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    )

    # SAM2 Model
    sam2_model = build_sam2(model_cfg, checkpoint_file, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

except Exception as e:
    logging.error(f"Error loading Models: {e}")
    raise

# Hand landmarks map for selecting correct label from yolo
finger_joints = {
    "Index": [5, 6],
    "Middle": [9, 10],
    "Ring": [13, 14],
    "Pinky": [17, 18],
}

# Finger Class Map
FINGER_CLASS_MAP = {
    0: 'Index',
    1: 'Middle',
    2: 'Pinky',
    3: 'Ring'
}

# Finger Color Map
FINGER_COLOR_MAP = {
    0: (255, 0, 0),     # Blue for Index finger
    1: (0, 255, 0),     # Green for Middle finger
    2: (0, 0, 255),     # Red for Pinky finger
    3: (255, 255, 0)    # Cyan for Ring finger
}

# Mediapipe hand landmarks map for SAM2
FINGER_JOINTS_FOR_MASK = {
    'Index': [6],
    'Middle': [10],
    'Pinky': [18],
    'Ring': [14]
}

options = vision.HandLandmarkerOptions(
    base_options=base_options, num_hands=1, min_hand_detection_confidence=0.10
)
detector = vision.HandLandmarker.create_from_options(options)


def background_removal(path):
    input_image = cv2.imread(path)
    foreground = remove(
        input_image,
        session=session,
        bgcolor=(255, 255, 255, 255),
        post_process_mask=True,
        alpha_matting=True,
        alpha_matting_foreground_threshold=280,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=12,
    )
    proper_foreground = cv2.cvtColor(foreground, code=cv2.COLOR_BGRA2BGR) #type:ignore
    return proper_foreground


def get_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)

    largest_contour = max(contours, key=cv2.contourArea)

    return largest_contour

def generate_polygon(contours, H, W):
    contours_normalized = contours.astype(np.float32)
    contours_normalized[:, :, 0] /= W
    contours_normalized[:, :, 1] /= H

    new_contours = contours_normalized * 100

    contour_tuple = np.vstack(new_contours).squeeze()

    final_polygon = [
        [round(float(x[0]), 2), round(float(x[1]), 2)] for x in contour_tuple
    ]

    # final_polygon = ", ".join(
    #     f"{round(float(x[0]), 2)}% {round(float(x[1]), 2)}%" for x in contour_tuple
    # )

    return final_polygon

def determine_handedness(landmarks: list[np.ndarray]) -> str:
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    pinky_mcp = landmarks[17]

    # Vector from wrist to middle MCP
    wrist_to_middle = middle_mcp - wrist

    # Vector from index MCP to pinky MCP
    index_to_pinky = pinky_mcp - index_mcp

    # Cross product to determine palm direction
    palm_normal = np.cross(wrist_to_middle, index_to_pinky)

    # Dot product to determine if palm is facing the camera
    palm_facing_camera = np.dot(wrist_to_middle, index_to_pinky) > 0

    # Determine handedness
    is_left = palm_normal > 0 if palm_facing_camera else palm_normal < 0

    return "left" if is_left else "right"

def get_binary_image(results, i):
    if results[0].masks is not None:
        mask_raw = results[0].masks[i].cpu().data.numpy().transpose(1, 2, 0)

        # Convert to single channel if mask_raw is not already
        if mask_raw.shape[-1] != 1:
            mask_raw = mask_raw[:, :, 0]

        h2, w2, _ = results[0].orig_img.shape
        mask = cv2.resize(mask_raw, (w2, h2))

        # Convert mask to 8-bit unsigned integer
        mask = (mask * 255).astype(np.uint8)

        # Smooth the mask using OpenCV
        smoothed_mask = cv2.medianBlur(mask, 37)  # Adjust kernel size as needed

        return smoothed_mask

# finger segmentation
def finger_segmentation(image_path):
    results = model.predict(
        image_path,
        max_det=9,
        conf=0.30,
        iou=0.5,
        classes=[0, 1, 2, 3],
        verbose=False,
        save=False,
        imgsz=640,
        rect=True,
        show_labels=False,
        show_boxes=False,
        retina_masks=False,
    )
    
    binary_images_dic = {}
    # logging.info(results[0].boxes)
    # logging.info("results[0].boxes.cls", results[0].boxes.cls)
    if results[0].masks is not None:
        for i in range(len(results[0].boxes.cls)): #type:ignore
            binary_images_dic[i] = get_binary_image(results, i)
        size = results[0].masks.data[0].to("cpu").numpy().shape

    else:
        raise ValueError("No fingers found in the image")
    
    # logging.info(f"Yolo result masks data[0]: {results[0].masks.data[0]}")
    # logging.info(f"Yolo mask type data[0]: {type(results[0].masks.data[0])}")
    return results, binary_images_dic, size

# mediapipe landmark recognition
def landmark_recognition(image_data):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_data)

    detection_result = detector.detect(image)

    return image, detection_result


def find_first_left_black_pixel(image, start_x, start_y, rotation_angle):
    x, y = start_x, start_y
    height, width = image.shape[:2]
    radian = math.radians(rotation_angle)
    while 0 <= x < width and 0 <= y < height:
        if image[int(y), int(x)] == 0:
            return int(x), int(y)
        x += math.cos(radian)
        y += math.sin(radian)
    return None

def find_first_right_black_pixel(image, start_x, start_y, rotation_angle):
    x, y = start_x, start_y
    height, width = image.shape[:2]
    radian = math.radians(rotation_angle)
    while 0 <= x < width and 0 <= y < height:
        if image[int(y), int(x)] == 0:
            return int(x), int(y)
        x -= math.cos(radian)
        y -= math.sin(radian)
    return None

def fingers_width_calculation(image_path):
    #Rembg time
    start2 = time.time()
    im = background_removal(image_path)
    end2 = time.time() - start2
    logging.info(f"Time taken by rembg: {end2}")
    # im = cv2.imread(image_path)
    start4 = time.time()
    results, binary_image_dic, size = finger_segmentation(im)
    end4 = time.time() - start4
    logging.info(f"Time taken by yolo model: {end4}")
    # logging.info(f"Binary image dict: {binary_image_dic}")
    # logging.info(f"size:{size}")

    # start6 = time.time()
    if binary_image_dic and size:
        image, detection_result = landmark_recognition(
            cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        )

        if detection_result.hand_landmarks:
            # finger_line_data = {}

            im_height, im_width = image.height, image.width
            hand_landmarks_denorm: list[np.ndarray] = [
                np.array([round(lm.x * im_width), round(lm.y * im_height)])
                for lm in detection_result.hand_landmarks[0]
            ]
            start3 = time.time()
            handedness = determine_handedness(hand_landmarks_denorm)
            # logging.info("handedness", handedness)
            end3 = time.time() - start3
            logging.info(f"Time taken by mediapipe to clean labels: {end3}")
            finger_names = list(finger_joints.keys())

            for _, v in binary_image_dic.items():
                for name in finger_names:
                    finger_points = finger_joints.get(name)

                    if finger_points:
                        joint_1 = detection_result.hand_landmarks[0][finger_points[0]]
                        joint_2 = detection_result.hand_landmarks[0][finger_points[1]]
                    x1, y1 = int((joint_1.x * im_width)), int(joint_1.y * im_height)
                    x2, y2 = int((joint_2.x * im_width)), int(joint_2.y * im_height)
                    xp = x2 + 0.3 * (x1 - x2)
                    yp = y2 + 0.3 * (y1 - y2)
                    dx = x2 - x1
                    dy = y2 - y1
                    angle_radians = math.atan2(dy, dx)
                    rotation_angle = math.degrees(angle_radians) - 90

            #         # getting polygon for mask
            #         contours = get_contours(v)
            #         polygon = generate_polygon(contours, im_height, im_width)

            #         end1 = find_first_left_black_pixel(v, xp, yp, rotation_angle)
            #         end2 = find_first_right_black_pixel(v, xp, yp, rotation_angle)
            #         if end1 and end2:
            #             n_end1 = (end1[0]) / (im_width), end1[1] / im_height
            #             n_end2 = (end2[0]) / (im_width), end2[1] / im_height

            #             if (end1[0] - end2[0]) == 0 and (end1[1] - end2[1]) == 0:
            #                 continue
            #             elif end1 is None or end2 is None:
            #                 continue
            #             else:
            #                 width_finger = round(
            #                     pow(
            #                         pow(n_end2[0] - n_end1[0], 2)
            #                         + pow(n_end2[1] - n_end1[1], 2),
            #                         0.5,
            #                     ),
            #                     3,
            #                 )
            #                 center = (
            #                     round((float(n_end1[0] + n_end2[0]) / 2), 3),
            #                     round((float(n_end1[1] + n_end2[1]) / 2), 3),
            #                 )
            #                 finger_line_data[name] = {
            #                     "left": n_end1,
            #                     "right": n_end2,
            #                     "width": width_finger,
            #                     "center": center,
            #                     "rotation_angle": rotation_angle,
            #                     "polygon": polygon,
            #                     "contour": contours
            #                 }
            #                 break

            # finger_line_data.update({"hand": handedness.capitalize()})
            # end6 = time.time() - start6
            # logging.info(f"Time taken for API Response generation: {end6}")

            return results, detection_result, handedness, image, finger_names

        if not detection_result.hand_landmarks:
            raise ValueError("No hand detected in the image")

def process_bboxes(bboxes, hand_landmarks=None, handedness=None):
    input_bboxes = []
    processed_classes = set()

    for bbox in bboxes:
        class_id = int(bbox.cls)
        confidence = bbox.conf.item()
        x_center, y_center, width, height = bbox.xyxyn[0]
        
        # Log each bbox details
        logging.info(f"Raw Bbox - Class: {class_id}, Conf: {confidence}, Coords: {x_center}, {y_center}, {width}, {height}")
        
        # Process finger classes once
        if class_id in [0, 1, 2, 3]: 
            # Ensuring only one bbox per class is selected
            if class_id not in processed_classes:
                input_bboxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                processed_classes.add(class_id)
    
    logging.info(f"Processed Input Bboxes: {input_bboxes}")
    return input_bboxes

def detect_hand_landmarks(image: np.ndarray) -> dict[int, list[list[int]]]:
    # Image shape
    H, W = image.shape[:2]

    # Conver image from BGR to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Image Processed
    results = hand_model.process(img_rgb)

    # Saving required landmarks
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]

        # Categorize landmarks by finger
        finger_landmarks = {} # Empty Dictionary
        for finger_idx, (finger_name, joint_indices) in enumerate(FINGER_JOINTS_FOR_MASK.items()):
            finger_landmarks[finger_idx] = [
                [int(landmarks.landmark[idx].x * W),
                 int(landmarks.landmark[idx].y * H)]
                 for idx in joint_indices
            ]
        return finger_landmarks
    return None # type:ignore

def load_bboxes(input_bboxes: List):
    bboxes = []
    for line in input_bboxes:
        parts = line.strip().split()
        class_index = int(parts[0])
        x_min = float(parts[1])
        y_min = float(parts[2])
        x_max = float(parts[3])
        y_max = float(parts[4])
        bboxes.append([class_index, x_min, y_min, x_max, y_max])
    # logging.info(f"bboxes:{bboxes}")
    # logging.info(type(bboxes))
    return bboxes

def normalize_bbox_to_pixels(bbox, image_width, image_height):
    class_index, x_min, y_min, x_max, y_max = bbox
    x_min_pixel = int(x_min * image_width)
    y_min_pixel = int(y_min * image_height)
    x_max_pixel = int(x_max * image_width)
    y_max_pixel = int(y_max * image_height)
    return [class_index, x_min_pixel, y_min_pixel, x_max_pixel, y_max_pixel]

def smooth_mask(mask, kernel_size=5):
    return cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
 
def convert_bboxes_to_dict(pixel_bboxes: List[List[int]]) -> Dict[int, list[int]]:
    return {bbox[0]: list(bbox[1:]) for bbox in pixel_bboxes}

def generate_masks_for_image(image_path: Path, predictor: SAM2ImagePredictor, bboxes: dict[int, list[int]], points: dict[int, list[list[int]]]) -> tuple[dict[str, dict], float]: #type:ignore
    try:
        start11 = time.time()
        start12 = time.time()
        # Read Image
        img_pil = Image.open(image_path)
        end12 = time.time() - start12
        logging.info(f" Time to read image in SAM2 pil command: {end12}")

        start13 = time.time()
        # Image for prediction
        predictor.set_image(img_pil)
        end13 = time.time() - start13
        logging.info(f"Time for set image: {end13}")

        sam_result = {}
        total_time = 0
        start10 = time.time()
        # For all fingers
        for finger_idx, finger_name in FINGER_CLASS_MAP.items():
            logging.info(f"Processing {finger_name} finger...")

            if finger_idx in points:
                landmark_points = np.array(points[finger_idx])
                landmark_points = landmark_points.reshape(-1, 2)
                logging.info(f"{finger_name} landmarks: {landmark_points}")

                # Convert to input format
                point_coords = landmark_points[:, np.newaxis, :] 
                point_labels = np.ones((len(point_coords), 1), dtype=int)

                logging.info(f"Point coords (converted): {point_coords}")
                logging.info(f"Point labels (converted): {point_labels}")
            
            if finger_idx in bboxes:
                finger_bbox = bboxes[finger_idx]
                finger_bbox = np.array(finger_bbox)    
                logging.info(f"{finger_name} bbox: {finger_bbox}")
                
                start_time = time.time()
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=finger_bbox,
                    multimask_output=False,
                )
                prediction_time = time.time() - start_time
                total_time += prediction_time
                # Store results for the finger
                sam_result[finger_name] = {
                    'masks': masks,
                    'scores': scores,
                    'bounding_boxes': finger_bbox,
                    'points': point_coords,
                }

                logging.info(f"Processed {finger_name} finger in {prediction_time:.2f} seconds")
            else:
                logging.warning(f"No points found for {finger_name} finger.")
            end10 = time.time()-start10
        logging.info(f"Total processing time for the hand: {total_time:.2f} seconds")
        logging.info(f"Time for SAM2 points and box processing: {end10}")
        end11 = time.time() - start11
        logging.info(f"Time for SAM2 check: {end11}")
        return sam_result, total_time

    except Exception as e:
        logging.info(f"Error generating masks: {e}")

def process_img(img_path: Path):
    prediction_times = []
    try:
        img_name = img_path.stem
        img = cv2.imread(str(img_path))
        img_new = img.copy()
        img_copy = img.copy()
        my_img = img.copy()
        img_height, img_width = img.shape[:2]
        logging.info(f"height: {img_height}, width: {img_width}")

        # Get finger landmarks
        start1 = time.time()
        try:
            finger_landmarks = detect_hand_landmarks(img)
            logging.info(f"Detected finger landmarks: {finger_landmarks}")
        except Exception as e:
            logging.error(f"Error in loading finger landmarks: {e}")
            raise
        end1 = time.time()-start1

        logging.info(f"Landmark points for sam2 time: {end1}")
        
        # Get bboxes
        try:
            # Get finger data for the image
            results_yolo, detection_result, handedness, image, finger_names = fingers_width_calculation(str(img_path)) #type:ignore
            start7 = time.time()
            if results_yolo[0].boxes is not None:
                bboxes = results_yolo[0].boxes

            # Generate bboxes
            bbox_input = process_bboxes(bboxes)

            new_bboxes = load_bboxes(bbox_input)
            pixel_bboxes = [normalize_bbox_to_pixels(bbox, img_width, img_height) for bbox in new_bboxes]
            logging.info(f"Pixel bounding boxes: {pixel_bboxes}")

            # Convert pixel_bboxes to dict[int, np.ndarray] format
            pixel_bboxes_dict = convert_bboxes_to_dict(pixel_bboxes)
            logging.info(f"Pixel bounding boxes (converted): {pixel_bboxes_dict}")
            end7 = time.time() - start7
            logging.info(f"Time taken for generating bboxes: {end7}")

        except Exception as e:
            logging.error(f"Error in processing bounding boxes: {e}")
        
        # getting SAM2 Results
        start9 = time.time()
        sam_results, time_taken = generate_masks_for_image(img_path, predictor, pixel_bboxes_dict, finger_landmarks)
        end9 = time.time() - start9
        logging.info(f"Time for SAM2: {end9}")
        prediction_times.append(time_taken)

        # Create Binary Image dictionary for all fingers
        binary_img_dict_sam2 = {}
        for finger_idx, finger_name in FINGER_CLASS_MAP.items():
            if finger_name in sam_results:
                # Convert mask to binary 
                binary_mask = (sam_results[finger_name]['masks'][0] > 0.5).astype(np.uint8)*255
                
                # Resize mask to image size if needed
                binary_mask_resized = cv2.resize(binary_mask, (img_width, img_height))
                
                binary_img_dict_sam2[finger_idx] = binary_mask_resized

        # logging.info(f"Binary image dict SAM2: {binary_img_dict_sam2}")

        # Size of SAM2 image mask
        for finger_name, sam_result in sam_results.items():
            masks = sam_result['masks']
            sam2_size = masks[0].shape 
            # break

        # logging.info(f"SAM2 mask size: {sam2_size}")

        try:
            start16 = time.time()
            if binary_img_dict_sam2 and sam2_size:

                if detection_result.hand_landmarks:
                    finger_line_data_sam2 = {}

                    im_height, im_width = image.height, image.width
                    hand_landmarks_denorm: list[np.ndarray] = [
                        np.array([round(lm.x * im_width), round(lm.y * im_height)])
                        for lm in detection_result.hand_landmarks[0]
                    ]

                    finger_names = list(finger_joints.keys())

                    for _, v in binary_img_dict_sam2.items():
                        for name in finger_names:
                            finger_points = finger_joints.get(name)

                            if finger_points:
                                joint_1 = detection_result.hand_landmarks[0][finger_points[0]]
                                joint_2 = detection_result.hand_landmarks[0][finger_points[1]]
                            x1, y1 = int((joint_1.x * im_width)), int(joint_1.y * im_height)
                            x2, y2 = int((joint_2.x * im_width)), int(joint_2.y * im_height)
                            xp = x2 + 0.3 * (x1 - x2)
                            yp = y2 + 0.3 * (y1 - y2)
                            dx = x2 - x1
                            dy = y2 - y1
                            angle_radians = math.atan2(dy, dx)
                            rotation_angle = math.degrees(angle_radians) - 90

                            # getting polygon for mask
                            contours = get_contours(v)
                            polygon = generate_polygon(contours, im_height, im_width)

                            end1 = find_first_left_black_pixel(v, xp, yp, rotation_angle)
                            end2 = find_first_right_black_pixel(v, xp, yp, rotation_angle)
                            if end1 and end2:
                                n_end1 = (end1[0]) / (im_width), end1[1] / im_height
                                n_end2 = (end2[0]) / (im_width), end2[1] / im_height

                                if (end1[0] - end2[0]) == 0 and (end1[1] - end2[1]) == 0:
                                    continue
                                elif end1 is None or end2 is None:
                                    continue
                                else:
                                    width_finger = round(
                                        pow(
                                            pow(n_end2[0] - n_end1[0], 2)
                                            + pow(n_end2[1] - n_end1[1], 2),
                                            0.5,
                                        ),
                                        3,
                                    )
                                    center = (
                                        round((float(n_end1[0] + n_end2[0]) / 2), 3),
                                        round((float(n_end1[1] + n_end2[1]) / 2), 3),
                                    )
                                    finger_line_data_sam2[name] = {
                                        "left": n_end1,
                                        "right": n_end2,
                                        "width": width_finger,
                                        "center": center,
                                        "rotation_angle": rotation_angle,
                                        "polygon": polygon,
                                        "contour": contours
                                    }
                                    break

                    finger_line_data_sam2.update({"hand": handedness.capitalize()})
                    end16 = time.time() - start16
                    logging.info(f"Time taken for API Response generation for SAM2 result: {end16}")

                if not detection_result.hand_landmarks:
                    raise ValueError("No hand detected in the image")

        except Exception as e:
            logging.error("Error processing image response: {e}")

        start8 = time.time()
        # Create a blank mask overlay for all fingers
        mask_overlay = np.zeros_like(img_copy, dtype=np.uint8)

        # For each finger
        for finger_name, sam_result in sam_results.items():
            masks = sam_result['masks']
            scores = sam_result['scores']
            bbox = sam_result['bounding_boxes']
            points = sam_result['points']

            # logging.info(f"SAM2 Masks: {masks}")
            # logging.info(f"SAM2 Mask tye: {type(masks)}")

            finger_idx = list(FINGER_CLASS_MAP.values()).index(finger_name)
            color = FINGER_COLOR_MAP[finger_idx]  # Assign color for the current finger
            logging.info(f"Assigning color {color} for {FINGER_CLASS_MAP[finger_idx]} finger")

            # Get bounding boxes and points
            x_min, y_min, x_max, y_max = bbox

            # Draw bounding box on the image
            cv2.rectangle(img_new, (x_min, y_min), (x_max, y_max), color, 2)

            # Draw landmark points on image
            for point in points:
                cv2.circle(img_new, tuple(point[0]), 3, color, -1)
            
            # Convert mask to binary 
            binary_mask = (masks[0] > 0.5).astype(np.uint8)*255
            # cv2.imwrite(str(bbox_dir/f'mask{finger_idx}.jpg'), binary_mask)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                cv2.drawContours(mask_overlay, contours, -1, color, -1)

        # Smooth the mask overlay
        smoothed_mask = smooth_mask(mask_overlay)

        # Blend the smoothed mask overlay with the original image
        alpha = 0.5  # Transparency level
        masked_image = cv2.addWeighted(img_copy, 1 - alpha, smoothed_mask.astype(np.uint8), alpha, 0)

        box_point_img_path = bbox_dir/f'bbox_pts_{img_name}.jpg'
        cv2.imwrite(str(box_point_img_path), img_new)
        logging.info(f"Saved image with points and bboxes: {box_point_img_path}")

        # Save the final image with all finger masks
        masked_image_path = mask_dir/f"masked_{img_name}.jpg"
        cv2.imwrite(str(masked_image_path), masked_image)
        logging.info(f"Saved image with all finger masks: {masked_image_path}")

        # Save the pure smoothed mask overlay for debugging
        mask_path = segments_dir/f"mask_{img_name}.jpg"
        cv2.imwrite(str(mask_path), smoothed_mask.astype(np.uint8))
        logging.info(f"Saved smoothed mask overlay: {mask_path}")

        logging.info(f"Successfully processed {img_name}")
        end8 = time.time() - start8
        logging.info(f"Time for generating mask and bbox images and saving it: {end8}")
            

        return finger_line_data_sam2, sam_results, binary_img_dict_sam2

    except Exception as e:
        logging.error(f"Error processing {img_name}: {str(e)}")

    if prediction_times:
        avg_time = sum(prediction_times) / len(prediction_times)
        logging.info(f"Average prediction time per image: {avg_time:.2f} seconds")
    else:
        logging.warning("No images processed.")

def check_finger_data(image_path: Path, finger_data: dict, results, detection_results):
    img = cv2.imread(str(image_path))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_name = image_path.stem
    my_img = img.copy()
    # my_img1 = img.copy()
    try:
        data = finger_data
        fingers = ["Index", "Middle", "Ring", "Pinky"]
        for i in range(len(fingers)):
            finger_data = data[fingers[i]]

            left_coords = finger_data["left"]
            right_coords = finger_data["right"]
            center_coords = finger_data["center"]
            rotation_angle = finger_data["rotation_angle"]
            polygon_coords = finger_data["polygon"]

            img_height, img_width = my_img.shape[:2]
            center_coords_rev = [(left_coords[0] + right_coords[0]) / 2, (left_coords[1] + right_coords[1]) / 2]
            left_pixel = (int(left_coords[0] * img_width), int(left_coords[1] * img_height))
            right_pixel = (int(right_coords[0] * img_width), int(right_coords[1] * img_height))
            center_pixel = (int(center_coords_rev[0] * img_width), int(center_coords_rev[1] * img_height))
            
            # Draw circles at the new coordinates
            cv2.circle(my_img, left_pixel, 2, (0, 0, 255), -1)
            cv2.circle(my_img, center_pixel, 2, (0, 0, 255), -1)
            cv2.circle(my_img, right_pixel, 2, (0, 0, 255), -1)

            polygon_pixels = [(int(coord[0] * img_width / 100), int(coord[1] * img_height / 100)) for coord in polygon_coords]
            cv2.polylines(my_img, [np.array(polygon_pixels)], isClosed=True, color=(0, 255, 0), thickness=1)

        mask_overlay = np.zeros_like(my_img, dtype=np.uint8)

        # For each finger
        for finger_name, sam_result in results.items():
            masks = sam_result['masks']
            # points = sam_result['points']
        
            # for point in points:
            #     cv2.circle(my_img, tuple(point[0]), 3, (255, 0, 0), -1)
            
            # Convert mask to binary 
            binary_mask = (masks[0] > 0.5).astype(np.uint8)*255
            # cv2.imwrite(str(bbox_dir/f'mask{finger_idx}.jpg'), binary_mask)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(mask_overlay, contours, -1, (255, 0, 0), -1)

         # Plot ALL MediaPipe hand landmarks
        if detection_results.hand_landmarks:
            # Get first hand's landmarks
            landmarks = detection_results.hand_landmarks[0]
            img_height, img_width = my_img.shape[:2]

         # Landmark groups
        landmark_groups = {
            'Wrist': [0],
            'Thumb': [1, 2, 3, 4],
            'Index': [5, 6, 7, 8],
            'Middle': [9, 10, 11, 12],
            'Ring': [13, 14, 15, 16],
            'Pinky': [17, 18, 19, 20]
        }

        # Add all hand landmark points
        for group_name, indices in landmark_groups.items():
            for idx in indices:
                x = int(landmarks[idx].x * img_width)
                y = int(landmarks[idx].y * img_height)
                cv2.circle(my_img, (x, y), 4, (255, 0, 0), -1)

        # Smooth the mask overlay
        smoothed_mask = smooth_mask(mask_overlay)

        # Blend the smoothed mask overlay with the original image
        alpha = 0.2  # Transparency level
        masked_image = cv2.addWeighted(my_img, 1 - alpha, smoothed_mask.astype(np.uint8), alpha, 0)

        check_finger_data_img_path = mask_dir/f'finger_data_{img_name}.jpg'
        cv2.imwrite(str(check_finger_data_img_path), my_img)
        logging.info(f"Saved Finger data image to: {check_finger_data_img_path}")

        # Save the final image with all finger masks
        masked_image_path_new = mask_dir/f"masked_finger_data_{img_name}.jpg"
        cv2.imwrite(str(masked_image_path_new), masked_image)
        logging.info(f"Saved image with all finger masks: {masked_image_path_new}")

    except Exception as e:
        logging.error(f"Error in processing finger data for {img_name}: {e}")

if __name__ == "__main__":
    total_time = 0
    total_files = 0
    image_file = "temp_image_01bde5c2-ba7c-4770-bc70-36970f60e2da.jpg"
    # for image_file in os.listdir(img_dir):
    #     if image_file.endswith('.jpg'):
    #         try:
    start5 = time.time()

    img_path = img_dir/image_file

    logging.info(f"Image path: {img_path}")

    # Process all images
    finger_data, _, _, results_new, detection_result = process_img(img_path) #type:ignore
    end5 = time.time() - start5
    logging.info(f"Total time taken for whole code: {end5}")
    total_time+=end5
    total_files+=1

    logging.info(25*"**")

    # logging.info(f"Finger data SAM2: {finger_data}")
    # logging.info(f"Finger data type: {type(finger_data)}")

    check_finger_data(img_path, finger_data, results_new, detection_result)

            # except Exception as e:
            #     logging.error(f"Error processing images: {e}")

    time_avg = total_time/total_files
    logging.info(f"Total images: {total_files}")
    logging.info(f"Average time taken at last final: {time_avg}")

