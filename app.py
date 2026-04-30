import os
import io
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageStat
import tensorflow as tf
import scipy.stats

app = Flask(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH           = 'skin_cancer_model.keras'
CONFIG_PATH          = 'skin_cancer_model_config.json'
IMG_SIZE             = 128
CONFIDENCE_THRESHOLD = 55.0
ENTROPY_THRESHOLD    = 1.9

# ── Class definitions ─────────────────────────────────────────────────────────
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

CLASS_INFO = {
    'MEL': {
        'full_name'  : 'Melanoma',
        'risk'       : 'HIGH',
        'risk_label' : 'High Risk',
        'color'      : '#ff4444',
        'icon'       : '⚠️',
        'description': 'Malignant melanoma is the most dangerous form of skin cancer. '
                       'Immediate dermatologist consultation is strongly recommended.',
        'action'     : 'Seek immediate medical attention'
    },
    'NV': {
        'full_name'  : 'Melanocytic Nevi',
        'risk'       : 'LOW',
        'risk_label' : 'Low Risk',
        'color'      : '#00c896',
        'icon'       : '✅',
        'description': 'Benign mole (common nevus). Generally harmless but should be '
                       'monitored for any changes in size, shape, or color.',
        'action'     : 'Monitor periodically'
    },
    'BCC': {
        'full_name'  : 'Basal Cell Carcinoma',
        'risk'       : 'MEDIUM',
        'risk_label' : 'Medium Risk',
        'color'      : '#ff9500',
        'icon'       : '⚡',
        'description': 'Most common form of skin cancer. Rarely spreads but requires '
                       'medical treatment. Schedule a dermatologist appointment.',
        'action'     : 'Schedule doctor visit'
    },
    'AKIEC': {
        'full_name'  : 'Actinic Keratosis',
        'risk'       : 'MEDIUM',
        'risk_label' : 'Medium Risk',
        'color'      : '#ff9500',
        'icon'       : '⚡',
        'description': 'Pre-cancerous lesion caused by UV damage. Can progress to '
                       'squamous cell carcinoma if untreated. Medical evaluation recommended.',
        'action'     : 'Medical evaluation needed'
    },
    'BKL': {
        'full_name'  : 'Benign Keratosis',
        'risk'       : 'LOW',
        'risk_label' : 'Low Risk',
        'color'      : '#00c896',
        'icon'       : '✅',
        'description': 'Benign (non-cancerous) skin lesion including seborrheic keratosis '
                       'and solar lentigo. Generally harmless.',
        'action'     : 'Routine monitoring'
    },
    'DF': {
        'full_name'  : 'Dermatofibroma',
        'risk'       : 'LOW',
        'risk_label' : 'Low Risk',
        'color'      : '#00c896',
        'icon'       : '✅',
        'description': 'Benign fibrous skin lesion. Usually firm and painless. '
                       'Very rarely associated with malignancy.',
        'action'     : 'No urgent action required'
    },
    'VASC': {
        'full_name'  : 'Vascular Lesions',
        'risk'       : 'LOW',
        'risk_label' : 'Low Risk',
        'color'      : '#00c896',
        'icon'       : '✅',
        'description': 'Benign vascular skin lesion such as angioma or hemangioma. '
                       'Caused by abnormal blood vessel growth.',
        'action'     : 'Consult if concerned'
    }
}

# ── Load Model ────────────────────────────────────────────────────────────────
print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    IMG_SIZE = cfg.get('img_size', IMG_SIZE)
    print(f"✅ Config loaded — IMG_SIZE={IMG_SIZE}")
else:
    print(f"⚠️  No config file found, using IMG_SIZE={IMG_SIZE}")


# ── Helper: Validate image quality ───────────────────────────────────────────
def is_valid_skin_image(image_bytes):
    """
    Basic image sanity checks before running the model.
    Returns (is_valid: bool, error_message: str or None)
    """
    try:
        img  = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        w, h = img.size

        # Too small
        if w < 50 or h < 50:
            return False, "Image is too small (minimum 50×50 pixels). Please upload a clear skin lesion photo."

        # Blank or solid color
        stat       = ImageStat.Stat(img)
        avg_stddev = sum(stat.stddev[:3]) / 3
        if avg_stddev < 8.0:
            return False, "Image appears blank or is a solid color. Please upload an actual skin lesion photo."

        # Extreme aspect ratio
        ratio = max(w, h) / min(w, h)
        if ratio > 5.0:
            return False, "Image has an unusual shape. Please upload a properly cropped skin lesion photo."

        # Too dark
        avg_mean = sum(stat.mean[:3]) / 3
        if avg_mean < 10:
            return False, "Image is too dark to analyze. Please upload a well-lit photo."

        # Too bright / overexposed
        if avg_mean > 245:
            return False, "Image is overexposed. Please upload a properly lit skin lesion photo."

        return True, None

    except Exception as e:
        return False, f"Could not read image file. Please try a different image. ({str(e)})"


# ── Helper: Preprocess image ──────────────────────────────────────────────────
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# ── Helper: Check prediction confidence ──────────────────────────────────────
def is_confident_prediction(probs):
    """
    Returns (is_confident: bool, error_message: str or None)
    """
    top_conf = float(np.max(probs)) * 100

    if top_conf < CONFIDENCE_THRESHOLD:
        return False, (
            f"This does not appear to be a skin lesion image. "
            f"The model confidence is too low ({top_conf:.1f}%). "
            f"Please upload a dermoscopy or skin lesion photo."
        )

    entropy = float(scipy.stats.entropy(probs))
    if entropy > ENTROPY_THRESHOLD:
        return False, (
            "The model could not identify this as a skin lesion. "
            "Please upload a clear dermoscopy or skin lesion photo."
        )

    return True, None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded on server.'}), 500

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected.'}), 400

    allowed = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
    ext     = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in allowed:
        return jsonify({'success': False, 'error': f'Invalid file type .{ext}. Allowed: JPG, PNG, BMP, WEBP.'}), 400

    try:
        image_bytes = file.read()

        # Step 1 — Image quality check
        valid, reason = is_valid_skin_image(image_bytes)
        if not valid:
            return jsonify({'success': False, 'rejected': True, 'message': reason})

        # Step 2 — Preprocess & predict
        arr   = preprocess_image(image_bytes)
        probs = model.predict(arr, verbose=0)[0]
        probs = np.array(probs).flatten()           # ensure shape is (7,)

        print(f"DEBUG — probs shape: {probs.shape}, values: {probs}")

        # Step 3 — Confidence check
        confident, reason = is_confident_prediction(probs)
        if not confident:
            return jsonify({'success': False, 'rejected': True, 'message': reason})

        # Step 4 — Build result
        pred_idx   = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        info       = CLASS_INFO[pred_class]
        confidence = round(float(probs[pred_idx]) * 100, 2)

        # Top 3
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [
            {
                'class'      : CLASS_NAMES[i],
                'full_name'  : CLASS_INFO[CLASS_NAMES[i]]['full_name'],
                'probability': round(float(probs[i]) * 100, 2),
                'color'      : CLASS_INFO[CLASS_NAMES[i]]['color']
            }
            for i in top3_idx
        ]

        # All probabilities
        all_probs = [
            {
                'class'      : c,
                'full_name'  : CLASS_INFO[c]['full_name'],
                'probability': round(float(p) * 100, 2),
                'color'      : CLASS_INFO[c]['color']
            }
            for c, p in zip(CLASS_NAMES, probs)
        ]

        print(f"DEBUG — predicted: {pred_class}, confidence: {confidence}%, top3: {[t['class'] for t in top3]}")

        return jsonify({
            'success'          : True,
            'predicted_class'  : pred_class,
            'full_name'        : info['full_name'],
            'confidence'       : confidence,
            'risk'             : info['risk'],
            'risk_label'       : info['risk_label'],
            'color'            : info['color'],
            'icon'             : info['icon'],
            'description'      : info['description'],
            'action'           : info['action'],
            'top3'             : top3,
            'all_probabilities': all_probs
        })

    except Exception as e:
        print(f"ERROR in /predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/health')
def health():
    return jsonify({
        'status'      : 'ok',
        'model_loaded': model is not None,
        'classes'     : CLASS_NAMES,
        'img_size'    : IMG_SIZE
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)