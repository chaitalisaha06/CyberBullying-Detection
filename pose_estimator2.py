# import os
# import torch
# import json
# import cv2
# import colorsys
# import numpy as np
# import matplotlib.colors as mcolors
# from PIL import Image, ImageDraw
# from torchvision import transforms

# ########################################
# # 1. Define label-to-ID mapping (if needed)
# ########################################
# LABELS_TO_IDS = {
#     "Background": 0,
#     "Apparel": 1,
#     "Face Neck": 2,
#     "Hair": 3,
#     "Left Foot": 4,
#     "Left Hand": 5,
#     "Left Lower Arm": 6,
#     "Left Lower Leg": 7,
#     "Left Shoe": 8,
#     "Left Sock": 9,
#     "Left Upper Arm": 10,
#     "Left Upper Leg": 11,
#     "Lower Clothing": 12,
#     "Right Foot": 13,
#     "Right Hand": 14,
#     "Right Lower Arm": 15,
#     "Right Lower Leg": 16,
#     "Right Shoe": 17,
#     "Right Sock": 18,
#     "Right Upper Arm": 19,
#     "Right Upper Leg": 20,
#     "Torso": 21,
#     "Upper Clothing": 22,
#     "Lower Lip": 23,
#     "Upper Lip": 24,
#     "Lower Teeth": 25,
#     "Upper Teeth": 26,
#     "Tongue": 27,
#     # ...
# }

# # Create an inverted dictionary:
# IDS_TO_LABELS = {v: k for k, v in LABELS_TO_IDS.items()}

# ########################################
# # 2. Load TorchScript Model
# ########################################
# model_scripted_path = r"D:\\Image Cyberbullying\\checkpoints\\pose\\sapiens_1b_coco_best_coco_AP_821_torchscript.pt2"

# model_scripted = torch.jit.load(model_scripted_path)
# model_scripted.eval()
# model_scripted.to("cuda")

# print("TorchScript model loaded successfully.")

# ########################################
# # 3. Utility Functions
# ########################################
# def resize_image(pil_image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
#     """
#     Resize a PIL image while maintaining its aspect ratio, then paste it
#     into a new image of size = target_size with black padding.
#     """
#     original_width, original_height = pil_image.size
#     target_width, target_height = target_size

#     # Calculate aspect ratios
#     aspect_ratio = original_width / original_height
#     target_aspect = target_width / target_height

#     if aspect_ratio > target_aspect:
#         # Image is wider than target, scale based on width
#         new_width = target_width
#         new_height = int(new_width / aspect_ratio)
#     else:
#         # Image is taller than target, scale based on height
#         new_height = target_height
#         new_width = int(new_height * aspect_ratio)

#     # Resize the image
#     resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

#     # Create a new image with the target size and paste the resized image
#     new_image = Image.new("RGB", target_size, (0, 0, 0))
#     paste_x = (target_width - new_width) // 2
#     paste_y = (target_height - new_height) // 2
#     new_image.paste(resized_image, (paste_x, paste_y))

#     return new_image


# def preprocess_image(image: np.ndarray, input_shape=(3, 1024, 768)) -> torch.Tensor:
#     """
#     Preprocess the raw image (as a NumPy array) to match the input format expected by the model.
#     """
#     # Resize via OpenCV
#     img = cv2.resize(
#         image,
#         (input_shape[2], input_shape[1]),
#         interpolation=cv2.INTER_LINEAR
#     ).transpose(2, 0, 1)  # HWC -> CHW

#     # Convert to Tensor and change BGR -> RGB (by swapping [2, 1, 0])
#     img = torch.from_numpy(img)
#     img = img[[2, 1, 0], ...].float()

#     # Normalization (custom mean and std used in your pipeline)
#     mean = torch.tensor([123.5, 116.5, 103.5]).view(-1, 1, 1)
#     std = torch.tensor([58.5, 57.0, 57.5]).view(-1, 1, 1)
#     img = (img - mean) / std

#     # Add batch dimension
#     return img.unsqueeze(0)


# def udp_decode(
#     heatmap: np.ndarray,
#     img_size: tuple[int, int],
#     heatmap_size: tuple[int, int]
# ):
#     """
#     Decode the heatmap into keypoints using a simplified UDP approach.
#     """
#     h, w = heatmap_size
#     num_joints = heatmap.shape[0]

#     keypoints = np.zeros((num_joints, 2))
#     keypoint_scores = np.zeros(num_joints)

#     for i in range(num_joints):
#         hm = heatmap[i]
#         idx = np.unravel_index(np.argmax(hm), hm.shape)  # (row, col) = (y, x)
#         # Convert to full image coordinates:
#         keypoints[i] = [idx[1] * img_size[1] / w,  # x
#                         idx[0] * img_size[0] / h]  # y
#         keypoint_scores[i] = hm[idx]

#     return keypoints, keypoint_scores


# @torch.inference_mode()
# def get_pose(
#     image: Image.Image,
#     pose_estimator: torch.jit.ScriptModule,
#     input_shape=(3, 1024, 768),
#     device="cuda"
# ):
#     """
#     Given a PIL image and a pose estimator model, returns the keypoints and their scores.
#     """
#     # Convert PIL image to a NumPy array (RGB)
#     np_image = np.array(image)

#     # If the model expects BGR, uncomment below:
#     # np_image = np_image[..., ::-1]

#     # Preprocess to shape (1, C, H, W)
#     img_tensor = preprocess_image(np_image, input_shape).to(device)

#     # Forward pass -> heatmap shape = [1, num_joints, H_out, W_out]
#     output_heatmap = pose_estimator(img_tensor)

#     # Convert to CPU numpy
#     heatmap_np = output_heatmap[0].cpu().numpy()

#     # Decode the heatmap (assuming H_out = input_shape[1]/4, W_out = input_shape[2]/4)
#     keypoints, keypoint_scores = udp_decode(
#         heatmap_np,
#         (input_shape[1], input_shape[2]),
#         (input_shape[1] // 4, input_shape[2] // 4)
#     )

#     return keypoints, keypoint_scores


# def visualize_keypoints(
#     image: Image.Image,
#     keypoints: np.ndarray,
#     keypoint_scores: np.ndarray,
#     threshold=0.3
# ) -> Image.Image:
#     """
#     Draw red circles on the image where the score is above threshold.
#     """
#     draw = ImageDraw.Draw(image)
#     for (x, y), score in zip(keypoints, keypoint_scores):
#         if score > threshold:
#             radius = 3
#             left_up = (x - radius, y - radius)
#             right_down = (x + radius, y + radius)
#             draw.ellipse([left_up, right_down], fill='red', outline='red')
#     return image


# def save_pose_to_json(keypoints: np.ndarray, keypoint_scores: np.ndarray, output_path: str):
#     """
#     Save the keypoints and their scores to a JSON file.
#     """
#     pose_data = []
#     for i, (kp, score) in enumerate(zip(keypoints, keypoint_scores)):
#         # Lookup the label name from the inverted dict
#         label_name = IDS_TO_LABELS.get(i, f"Unknown_{i}")
#         pose_data.append({
#             "id": label_name,
#             "x": float(kp[0]),
#             "y": float(kp[1]),
#             "score": float(score)
#         })

#     with open(output_path, 'w') as f:
#         json.dump(pose_data, f, indent=4)

#     print(f"Pose data saved to: {output_path}")

# ########################################
# # 4. Main Execution
# ########################################
# if __name__ == "__main__":
#     # Directories
#     image_dir = r"D:\Image Cyberbullying\images\offensive images"
#     json_output_dir = os.path.join(image_dir, "JSON files of Images")

#     # Ensure the JSON output directory exists
#     os.makedirs(json_output_dir, exist_ok=True)

#     # Valid image extensions
#     valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

#     # Loop through all files in image_dir
#     for filename in os.listdir(image_dir):
#         # Check if it's an image file
#         if filename.lower().endswith(valid_exts):
#             image_path = os.path.join(image_dir, filename)
#             print(f"Processing: {image_path}")

#             # Load the local image
#             pil_image = Image.open(image_path)
#             if pil_image.mode == "RGBA":
#                 pil_image = pil_image.convert("RGB")

#             # Optionally resize the image before feeding to model
#             resized_pil_image = resize_image(pil_image, (640, 480))

#             # Pose estimation
#             keypoints, keypoint_scores = get_pose(
#                 resized_pil_image,
#                 model_scripted,
#                 input_shape=(3, 1024, 768),
#                 device="cuda"
#             )

#             # Visualize (optional)
#             # vis_image = visualize_keypoints(resized_pil_image.copy(), keypoints, keypoint_scores)
#             # vis_image.show() or save -> vis_image.save(f"{os.path.splitext(filename)[0]}_pose.jpg")

#             # Save the JSON with the same base name
#             base_name, _ = os.path.splitext(filename)
#             json_output_path = os.path.join(json_output_dir, f"{base_name}.json")
#             save_pose_to_json(keypoints, keypoint_scores, json_output_path)

# # Visualize keypoints on the image
# vis_image = visualize_keypoints(resized_pil_image.copy(), keypoints, keypoint_scores, threshold=0.3)

# # Save the visualized image
# vis_output_path = os.path.join(image_dir, f"{base_name}_pose_visualized.jpg")
# vis_image.save(vis_output_path)
# print(f"Pose visualized image saved to: {vis_output_path}")

# # Optionally display the visualized image
# vis_image.show()
import os
import torch
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Load the pre-trained model
model_scripted_path = r"D:\\Image Cyberbullying\\checkpoints\\pose\\sapiens_1b_coco_best_coco_AP_821_torchscript.pt2"
model_scripted = torch.jit.load(model_scripted_path)
model_scripted.eval()
model_scripted.to("cuda")

# Function to resize image while maintaining aspect ratio
def resize_image(pil_image: Image.Image, target_size=(640, 480)) -> Image.Image:
    original_width, original_height = pil_image.size
    aspect_ratio = original_width / original_height
    target_width, target_height = target_size

    if aspect_ratio > target_width / target_height:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    new_image = Image.new("RGB", target_size, (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image

# Function to preprocess image for the model
def preprocess_image(image: np.ndarray, input_shape=(3, 1024, 768)) -> torch.Tensor:
    img = cv2.resize(image, (input_shape[2], input_shape[1]), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
    img = torch.from_numpy(img)[[2, 1, 0], ...].float()
    mean = torch.tensor([123.5, 116.5, 103.5]).view(-1, 1, 1)
    std = torch.tensor([58.5, 57.0, 57.5]).view(-1, 1, 1)
    img = (img - mean) / std
    return img.unsqueeze(0)

# Function to decode keypoints
def udp_decode(heatmap: np.ndarray, img_size=(1024, 768), heatmap_size=(256, 192)):
    h, w = heatmap_size
    num_joints = heatmap.shape[0]
    keypoints = np.zeros((num_joints, 2))
    keypoint_scores = np.zeros(num_joints)

    for i in range(num_joints):
        hm = heatmap[i]
        idx = np.unravel_index(np.argmax(hm), hm.shape)
        keypoints[i] = [idx[1] * img_size[1] / w, idx[0] * img_size[0] / h]
        keypoint_scores[i] = hm[idx]

    return keypoints, keypoint_scores

# Function to get pose estimation
@torch.inference_mode()
def get_pose(image: Image.Image, pose_estimator: torch.jit.ScriptModule, input_shape=(3, 1024, 768), device="cuda"):
    np_image = np.array(image)
    img_tensor = preprocess_image(np_image, input_shape).to(device)
    output_heatmap = pose_estimator(img_tensor)
    heatmap_np = output_heatmap[0].cpu().numpy()
    keypoints, keypoint_scores = udp_decode(heatmap_np, (input_shape[1], input_shape[2]), (input_shape[1] // 4, input_shape[2] // 4))
    return keypoints, keypoint_scores

# Function to save keypoints as JSON with sequential names
def save_pose_to_json(keypoints: np.ndarray, keypoint_scores: np.ndarray, output_dir: str, index: int):
    output_path = os.path.join(output_dir, f"image{index}.json")
    pose_data = [{"id": i, "x": float(kp[0]), "y": float(kp[1]), "score": float(score)} for i, (kp, score) in enumerate(zip(keypoints, keypoint_scores))]

    with open(output_path, 'w') as f:
        json.dump(pose_data, f, indent=4)

    print(f"Pose data saved to: {output_path}")

# Main execution
if __name__ == "__main__":
    image_dir = r"D:\Image Cyberbullying\images\Test image"
    json_output_dir = os.path.join(image_dir, "JSON files of Images")

    os.makedirs(json_output_dir, exist_ok=True)

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]
    image_files.sort()  # Ensure sequential processing

    for idx, filename in enumerate(image_files, start=1):
        image_path = os.path.join(image_dir, filename)
        print(f"Processing: {image_path}")

        try:
            pil_image = Image.open(image_path)
            if pil_image.mode == "RGBA":
                pil_image = pil_image.convert("RGB")

            resized_pil_image = resize_image(pil_image, (640, 480))

            keypoints, keypoint_scores = get_pose(resized_pil_image, model_scripted)

            save_pose_to_json(keypoints, keypoint_scores, json_output_dir, idx)

        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")
