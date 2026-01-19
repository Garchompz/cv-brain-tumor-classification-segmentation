from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from PIL import Image
import io
import base64
import cv2
from datetime import datetime
import os
from skimage.feature import graycomatrix, graycoprops # Import wajib untuk ekstraksi fitur
from tqdm import tqdm

app = Flask(__name__)
CORS(app)

# --- KONFIGURASI ---
# Sesuaikan path model kamu
MODEL_PATH = 'D:\APP AOL\Computer Vision/app/backend/rf_seg_simple.joblib'
PATCH_SIZE = 7 # Harus SAMA dengan saat training

model = None

# --- LOAD MODEL ---
try:
    # Gunakan mmap_mode='r' agar hemat RAM
    model = joblib.load(MODEL_PATH, mmap_mode='r')
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {str(e)}")

# --- FUNGSI FEATURE EXTRACTION (Sama persis dengan Training) ---
def glcm_features(patch, levels=16): # levels harus sama dengan training
    # Normalisasi patch ke integer range 0-15 (untuk GLCM)
    if patch.dtype.kind == 'f':
        patch_scaled = np.clip(np.round(patch * (levels - 1)), 0, levels - 1).astype(np.uint8)
    else:
        patch_scaled = np.clip(patch, 0, levels - 1).astype(np.uint8)
    
    glcm = graycomatrix(
        patch_scaled,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=levels,
        symmetric=True,
        normed=True
    )
    
    # Entropy calculation
    eps = 1e-10
    p = glcm / (glcm.sum(axis=(0,1), keepdims=True) + eps)
    entropies = -np.sum(np.where(p > 0, p * np.log2(p), 0.0), axis=(0,1))
    entropy_mean = entropies.mean()
    
    feats = {
        'contrast': graycoprops(glcm, 'contrast').mean(),
        'correlation': graycoprops(glcm, 'correlation').mean(),
        'energy': graycoprops(glcm, 'energy').mean(),
        'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
        'entropy': entropy_mean
    }
    return feats

def extract_patch_features(img, x, y, patch_size=7):
    # Ambil patch dari koordinat x,y
    patch = img[x:x+patch_size, y:y+patch_size]
    
    # Intensity stats
    mean_intensity = patch.mean()
    var_intensity  = patch.var()
    
    # GLCM features
    glcm_feats = glcm_features(patch)
    
    # Urutan fitur HARUS SAMA dengan saat training
    feature_vector = [
        mean_intensity,
        var_intensity,
        glcm_feats['contrast'],
        glcm_feats['correlation'],
        glcm_feats['energy'],
        glcm_feats['homogeneity'],
        glcm_feats['entropy']
    ]
    
    return feature_vector

# --- FUNGSI UTILITAS LAINNYA ---
def image_to_base64(image):
    if len(image.shape) == 2:
        pil_image = Image.fromarray(image.astype(np.uint8))
    else:
        pil_image = Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

def overlay_mask(original_image, mask, alpha=0.5):
    # Convert original to RGB if grayscale
    if len(original_image.shape) == 2:
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = original_image.copy()

    mask = (mask > 0).astype(np.uint8)

    if np.sum(mask) == 0:
        return original_rgb

    colored_mask = np.zeros_like(original_rgb)
    colored_mask[:, :, 0] = 255  # RED

    mask_bool = mask.astype(bool)

    overlayed = original_rgb.copy()
    overlayed[mask_bool] = cv2.addWeighted(original_rgb[mask_bool],1 - alpha,colored_mask[mask_bool],alpha,0)

    return overlayed


def segment_image_proba(img, rf_model, patch_size=7): # function for segmenting a full image
    h, w = img.shape
    half = patch_size // 2
    padded_img = np.pad(img, pad_width=half, mode='reflect')

    features = []
    coords = []

    # Extract features for every pixel
    for x in tqdm(range(h), total=h, desc="Extracting features for segmentation"):
        for y in range(w):
            feats = extract_patch_features(padded_img, x+half, y+half, patch_size)
            features.append(feats)
            coords.append((x,y))

    print("Feature extraction completed. Predicting labels...")

    # Predict labels
    proba_map = rf_model.predict_proba(features)[:,1].reshape(h, w)
    smooth_map = cv2.GaussianBlur(proba_map, (5,5), 2) # smoothen probability map
    binary_mask = (smooth_map > 0.5).astype(np.uint8)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    if stats.shape[0] > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final_mask = (labels == largest).astype(np.uint8)
    else:
        final_mask = np.zeros((h, w), dtype=np.uint8)

    # Reconstruct mask
    mask_pred = np.zeros((h,w), dtype=np.uint8)
    for (x,y), p in zip(coords, final_mask.flatten()):
        mask_pred[x,y] = p

    return mask_pred

# --- ROUTES ---
@app.route('/')
def home():
    return jsonify({"status": "running", "model": model is not None})

@app.route('/segment', methods=['POST'])
def segment():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    
    try:
        # 1. Baca Gambar
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L') # Convert ke Grayscale langsung
        img_original = np.array(pil_image)
        
        # 2. Resize agar proses cepat (Sesuai target size training, misal 128 atau 256)
        # Hati-hati: Jika training pake 256, sebaiknya disini 256 juga.
        # Kita pakai 128 untuk kecepatan demo.
        TARGET_SIZE = (256, 256)
        img_resized = cv2.resize(img_original, TARGET_SIZE)
        
        # 3. Normalisasi Z-Score (SAMA dengan Training)
        mean = np.mean(img_resized)
        std = np.std(img_resized)
        img_z = (img_resized - mean) / std
            
        # Min-Max Scaling ke 0-1
        img_norm = cv2.normalize(img_z, None, 0, 1, cv2.NORM_MINMAX)
        
        # 7. Reshape kembali ke bentuk gambar
        mask_pred = segment_image_proba(img_norm, model, PATCH_SIZE)
        
        # 9. Resize hasil kembali ke ukuran asli gambar upload (untuk display)
        orig_h, orig_w = img_original.shape
        mask_final = cv2.resize(mask_pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        mask_vis = (mask_final * 255).astype(np.uint8)

        
        # 10. Buat Overlay
        img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_GRAY2RGB)
        segmented_overlay = overlay_mask(img_original_rgb, mask_final)

        # Hitung statistik
        tumor_pixels = np.sum(mask_final > 0)
        total_pixels = mask_final.size
        tumor_percentage = (tumor_pixels / total_pixels) * 100

        return jsonify({
            "success": True,
            "data": {
                "mask": image_to_base64(mask_vis),
                "segmented": image_to_base64(segmented_overlay),
                "statistics": {
                    "tumor_percentage": round(tumor_percentage, 2),
                    "tumor_pixels": int(tumor_pixels),
                    "total_pixels": int(total_pixels)
                }
            }
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000,use_reloader=False)