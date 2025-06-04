import cv2
import numpy as np
from skimage.color import rgb2lab
import os
import pandas as pd

def compute_gcf(image):
    if image is None or image.size == 0:
        raise ValueError("Invalid image input")
    
    lab_image = rgb2lab(image)
    l_hist, _ = np.histogram(lab_image[:, :, 0], bins=256, range=(0, 100))
    a_hist, _ = np.histogram(lab_image[:, :, 1], bins=256, range=(-128, 127))
    b_hist, _ = np.histogram(lab_image[:, :, 2], bins=256, range=(-128, 127))
    
    l_hist = l_hist / (l_hist.sum() or 1)  # Avoid division by zero
    a_hist = a_hist / (a_hist.sum() or 1)
    b_hist = b_hist / (b_hist.sum() or 1)
    
    uniform_l = np.ones_like(l_hist) / len(l_hist)
    uniform_a = np.ones_like(a_hist) / len(a_hist)
    uniform_b = np.ones_like(b_hist) / len(b_hist)
    
    l_fidelity = np.linalg.norm(l_hist - uniform_l)
    a_fidelity = np.linalg.norm(a_hist - uniform_a)
    b_fidelity = np.linalg.norm(b_hist - uniform_b)
    
    gcf= round(l_fidelity + a_fidelity + b_fidelity, 3)
    

    return gcf


def compute_gcc(image):
    if image is None or image.size == 0:
        raise ValueError("Invalid image input")
    
    lab = rgb2lab(image)
    window_size = 8
    
    # Check if image is too small for window size
    if lab.shape[0] < window_size or lab.shape[1] < window_size:
        raise ValueError(f"Image too small for {window_size}x{window_size} window")
    
    local_contrasts = []
    for i in range(0, lab.shape[0] - window_size + 1, window_size):
        for j in range(0, lab.shape[1] - window_size + 1, window_size):
            window = lab[i:i+window_size, j:j+window_size]
            l_contrast = np.std(window[:, :, 0]) or 1e-6  # Avoid zero std
            a_contrast = np.std(window[:, :, 1]) or 1e-6
            b_contrast = np.std(window[:, :, 2]) or 1e-6
            local_contrasts.append([l_contrast, a_contrast, b_contrast])
    
    if not local_contrasts:
        raise ValueError("No valid windows found in image")
        
    local_contrasts = np.array(local_contrasts)
    means = np.mean(local_contrasts, axis=0)
    stds = np.std(local_contrasts, axis=0)
    consistency = 1 - stds / (means + 1e-6)  # Avoid division by zero
    
    return round(np.sum(consistency * [0.6, 0.2, 0.2]), 3)

def get_image_metrics():
    base_path = r"D:\Image Cyberbullying\images\Test image"
    valid_extensions = {'.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.JPEG','.webp'}
    rows = []
    
    for image_name in os.listdir(base_path):
        if os.path.splitext(image_name)[1].lower() in valid_extensions:
            try:
                image_path = os.path.join(base_path, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Failed to load image: {image_name}")
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rows.append({
                    "Image Name": image_name,
                    "GCF": compute_gcf(image),
                    "GCC": compute_gcc(image)
                })
            except Exception as e:
                print(f'Error processing {image_name}: {str(e)}')
    
    metrics_df = pd.DataFrame(rows)
    return metrics_df.set_index("Image Name")

if __name__ == "__main__":
    metrics_df = get_image_metrics()
    metrics_df.to_csv("image_metrics_test.csv")