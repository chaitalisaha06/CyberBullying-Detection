import os
import torch
import numpy as np
import cv2
from PIL import Image
from skimage.color import rgb2lab
import joblib
import pandas as pd
from tensorflow import keras
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer

# ==================== IMAGE PREPROCESSING ====================
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

def preprocess_image(image: np.ndarray, input_shape=(3, 1024, 768)) -> torch.Tensor:
    img = cv2.resize(image, (input_shape[2], input_shape[1]), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
    img = torch.from_numpy(img)[[2, 1, 0], ...].float()
    mean = torch.tensor([123.5, 116.5, 103.5]).view(-1, 1, 1)
    std = torch.tensor([58.5, 57.0, 57.5]).view(-1, 1, 1)
    img = (img - mean) / std
    return img.unsqueeze(0)

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

@torch.inference_mode()
def get_pose(image: Image.Image, pose_estimator: torch.jit.ScriptModule, input_shape=(3, 1024, 768), device="cuda"):
    np_image = np.array(image)
    img_tensor = preprocess_image(np_image, input_shape).to(device)
    output_heatmap = pose_estimator(img_tensor)
    heatmap_np = output_heatmap[0].cpu().numpy()
    return udp_decode(
        heatmap_np,
        (input_shape[1], input_shape[2]),
        (input_shape[1] // 4, input_shape[2] // 4)
    )

# ==================== IMAGE QUALITY METRICS ====================
def compute_gcf(image):
    """Compute Global Contrast Factor"""
    lab_image = rgb2lab(image)
    l_hist, _ = np.histogram(lab_image[:, :, 0], bins=256, range=(0, 100))
    a_hist, _ = np.histogram(lab_image[:, :, 1], bins=256, range=(-128, 127))
    b_hist, _ = np.histogram(lab_image[:, :, 2], bins=256, range=(-128, 127))
    
    l_hist = l_hist / (l_hist.sum() or 1)
    a_hist = a_hist / (a_hist.sum() or 1)
    b_hist = b_hist / (b_hist.sum() or 1)
    
    uniform_l = np.ones_like(l_hist) / len(l_hist)
    uniform_a = np.ones_like(a_hist) / len(a_hist)
    uniform_b = np.ones_like(b_hist) / len(b_hist)
    
    l_fidelity = np.linalg.norm(l_hist - uniform_l)
    a_fidelity = np.linalg.norm(a_hist - uniform_a)
    b_fidelity = np.linalg.norm(b_hist - uniform_b)
    
    gcf = round(l_fidelity + a_fidelity + b_fidelity, 3)
    return gcf

def compute_gcc(image):
    lab = rgb2lab(image)
    window_size = 8
    if lab.shape[0] < window_size or lab.shape[1] < window_size:
        raise ValueError("Image too small for GCC window size")
    local_contrasts = []
    for i in range(0, lab.shape[0] - window_size + 1, window_size):
        for j in range(0, lab.shape[1] - window_size + 1, window_size):
            window = lab[i:i+window_size, j:j+window_size]
            local_contrasts.append([
                np.std(window[:, :, 0]) or 1e-6,
                np.std(window[:, :, 1]) or 1e-6,
                np.std(window[:, :, 2]) or 1e-6
            ])
    local_contrasts = np.array(local_contrasts)
    means = np.mean(local_contrasts, axis=0)
    stds = np.std(local_contrasts, axis=0)
    consistency = 1 - stds / (means + 1e-6)
    return round(np.sum(consistency * [0.6, 0.2, 0.2]), 3)

# ==================== LANGUAGE-VISION MODELS ====================
def load_vision_language_models(use_gpu=True):
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    dtype = torch.float16 if use_gpu else torch.float32
    print("Loading LLaVA...")
    model_id = "llava-hf/llava-1.5-7b-hf"
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, load_in_4bit=use_gpu).to(device)
    llava_processor = AutoProcessor.from_pretrained(model_id)

    print("Loading FLAN-T5...")
    flan_model_id = "google/flan-t5-small"
    flan_tokenizer = AutoTokenizer.from_pretrained(flan_model_id)
    flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_id).to(device)

    return llava_model, llava_processor, flan_model, flan_tokenizer, device

def get_llava_description(image_path, llava_model, llava_processor, device):
    conversation = [{
        "role": "user",
        "content": [{"type": "text", "text": "Tell me about the image"}, {"type": "image"}]
    }]
    prompt = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
    image = Image.open(image_path).convert("RGB")
    inputs = llava_processor(images=image, text=prompt, return_tensors='pt').to(device)
    output = llava_model.generate(**inputs, max_new_tokens=200)
    return llava_processor.decode(output[0][2:], skip_special_tokens=True)

def generate_summary(prompt, flan_model, flan_tokenizer, device):
    input_text = f"Explain why the image is offensive. {prompt}" if "offensive" in prompt.lower() \
                 else f"Summarize the image: {prompt}"
    inputs = flan_tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    outputs = flan_model.generate(inputs["input_ids"], max_length=150, temperature=0.5)
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==================== MAIN INFERENCE FUNCTION ====================
def process_single_image(image_path, model_paths):
    print(f"\nProcessing: {image_path}")
    pose_model = torch.jit.load(model_paths["pose_model"]).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_model.to(device)

    pil_image = Image.open(image_path).convert("RGB")
    resized = resize_image(pil_image, (640, 480))
    keypoints, scores = get_pose(resized, pose_model, device=device)

    keypoint_data = {}
    for i, (kp, sc) in enumerate(zip(keypoints, scores)):
        keypoint_data[f"{i}_x"] = kp[0]
        keypoint_data[f"{i}_y"] = kp[1]
        keypoint_data[f"{i}_score"] = sc

    image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    gcf = compute_gcf(image_rgb)
    gcc = compute_gcc(image_rgb)
    keypoint_data["GCF"] = gcf
    keypoint_data["GCC"] = gcc
    df = pd.DataFrame([keypoint_data])

    scaler = joblib.load(model_paths["scaler"])
    pca = joblib.load(model_paths["pca"])
    model = keras.models.load_model(model_paths["nn_model"])
    transformed = pca.transform(scaler.transform(df))
    prob = model.predict(transformed).flatten()[0]
    prediction = "Offensive" if prob >= 0.5 else "Not Offensive"

    # Vision-language explanation
    llava_model, llava_proc, flan_model, flan_tok, vl_device = load_vision_language_models()
    description = get_llava_description(image_path, llava_model, llava_proc, vl_device)
    summary = generate_summary(description, flan_model, flan_tok, vl_device)

    return {
        "prediction": prediction,
        "probability_offensive": round(float(prob), 3),
        "gcf": gcf,
        "gcc": gcc,
        "llava_description": description,
        "summary": f"This image is {prediction} because: {summary}"
    }

# ==================== Example Usage ====================
if __name__ == "__main__":
    test_image_path = r"D:\Image Cyberbullying\images\Test image\Offensive_Guy_Showing_The_Middle_Finger_In_Front_Of_Camera_And_Being_Impolite_high_resolution_preview_3062149.jpg"
    model_paths = {
        "pose_model": "D:\\Image Cyberbullying\\checkpoints\\pose\\sapiens_1b_coco_best_coco_AP_821_torchscript.pt2",
        "scaler": "scaler.pkl",
        "pca": "pca.pkl",
        "nn_model": "neural_network_trained_model1.h5"
    }

    result = process_single_image(test_image_path, model_paths)
    for k, v in result.items():
        print(f"{k}: {v}")
