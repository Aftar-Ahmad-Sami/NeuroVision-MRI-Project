import os
import io
import base64
import tempfile
import numpy as np
from PIL import Image
import matplotlib.cm as cm # Added for Heatmap coloring

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import timm

# File Format Libraries
import pydicom 
import scipy.io 
import h5py 

from flask import Flask, render_template, request, jsonify

# ============================================================================

class Config:
    MODEL_TUMOR_CLASS = 'models/efficientnet_b0_best.pth'
    MODEL_PLANAR = 'models/plane_classifier.pth'
    
    MODEL_SEGMENTATION = {
        'axial': 'models/ax_best_model.pth',
        'coronal': 'models/co_best_model.pth',
        'sagittal': 'models/sa_best_model.pth'
    }

    TUMOR_CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    PLANAR_CLASSES = ['axial', 'coronal', 'sagittal'] 
    
    IMG_SIZE_CLS = 224
    IMG_SIZE_SEG = 512
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# MODELS
# ============================================================================

class PlaneClassifier(nn.Module):
    def __init__(self):
        super(PlaneClassifier, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 3)
        )

    def forward(self, x):
        return self.model(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.5))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = F.interpolate(self.global_avg_pool(x), size=x4.size()[2:], mode='bilinear', align_corners=False)
        return self.project(torch.cat((x1, x2, x3, x4, x5), dim=1))

class DeepLabHead(nn.Module):
    def __init__(self, n_classes, low_level_channels=256, aspp_in_channels=2048, decoder_channels=256):
        super(DeepLabHead, self).__init__()
        self.project_low_level = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.aspp = ASPP(aspp_in_channels, decoder_channels)
        self.classifier = nn.Sequential(
            nn.Conv2d(decoder_channels + 48, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1)
        )

    def forward(self, x_low, x_high):
        x_aspp = F.interpolate(self.aspp(x_high), size=x_low.size()[2:], mode='bilinear', align_corners=False)
        x_low = self.project_low_level(x_low)
        return self.classifier(torch.cat([x_aspp, x_low], dim=1))

class DeepLabV3Plus(nn.Module):
    def __init__(self, n_channels=1, n_classes=2):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = models.resnet50(weights=None, replace_stride_with_dilation=[False, False, True])
        self.backbone.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_stem = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool)
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4
        self.head = DeepLabHead(n_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.encoder_stem(x)
        x_low = self.layer1(x)
        x = self.layer2(x_low)
        x = self.layer3(x)
        x_high = self.layer4(x)
        x = self.head(x_low, x_high)
        return F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

# ============================================================================
# APP SETUP
# ============================================================================

app = Flask(__name__)
app.secret_key = 'bme-mri-project-secret'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

models_registry = {
    'classifier': None,
    'planar': None,
    'seg_axial': None,
    'seg_coronal': None,
    'seg_sagittal': None
}

def load_models():
    print("⏳ Loading Models into Memory...")
    
    if os.path.exists(Config.MODEL_TUMOR_CLASS):
        # Using timm EfficientNet
        m = timm.create_model('efficientnet_b0', pretrained=False, num_classes=4)
        ckpt = torch.load(Config.MODEL_TUMOR_CLASS, map_location=Config.DEVICE)
        if 'model_state_dict' in ckpt: ckpt = ckpt['model_state_dict']
        m.load_state_dict(ckpt)
        m.to(Config.DEVICE).eval()
        models_registry['classifier'] = m
        print("✅ Tumor Classifier Loaded")
    
    if os.path.exists(Config.MODEL_PLANAR):
        m = PlaneClassifier()
        ckpt = torch.load(Config.MODEL_PLANAR, map_location=Config.DEVICE)
        if 'model_state_dict' in ckpt: ckpt = ckpt['model_state_dict']
        elif 'state_dict' in ckpt: ckpt = ckpt['state_dict']
        m.load_state_dict(ckpt)
        m.to(Config.DEVICE).eval()
        models_registry['planar'] = m
        print("✅ Planar Classifier Loaded")

    for plane, path in Config.MODEL_SEGMENTATION.items():
        if os.path.exists(path):
            m = DeepLabV3Plus(n_channels=1, n_classes=2)
            ckpt = torch.load(path, map_location=Config.DEVICE)
            if 'model_state_dict' in ckpt: ckpt = ckpt['model_state_dict']
            m.load_state_dict(ckpt)
            m.to(Config.DEVICE).eval()
            models_registry[f'seg_{plane}'] = m
            print(f"✅ Segmentation Model ({plane}) Loaded")

with app.app_context():
    load_models()

# Transforms
trans_tumor = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE_CLS, Config.IMG_SIZE_CLS)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trans_gray_planar = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE_CLS, Config.IMG_SIZE_CLS)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

trans_gray_seg = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE_SEG, Config.IMG_SIZE_SEG)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ============================================================================
# HELPERS
# ============================================================================

def normalize_and_convert_to_png(data):
    data = data.astype(float)
    data = (np.maximum(data, 0) / data.max()) * 255.0
    data = np.uint8(data)
    
    img = Image.fromarray(data)
    buff = io.BytesIO()
    img.save(buff, format="PNG")
    return buff.getvalue()

def convert_dicom_to_png_bytes(dicom_bytes):
    try:
        ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
        return normalize_and_convert_to_png(ds.pixel_array)
    except Exception as e:
        raise ValueError(f"Failed to process DICOM: {str(e)}")

def convert_mat_to_png_bytes(mat_bytes):
    try:
        mat = scipy.io.loadmat(io.BytesIO(mat_bytes))
        image_data = None
        if 'cjdata' in mat:
            cjdata = mat['cjdata']
            if 'image' in cjdata.dtype.names:
                image_data = cjdata['image'][0, 0]
        if image_data is None:
            for key in ['image', 'img', 'data', 'IM', 'im']:
                if key in mat:
                    image_data = mat[key]
                    break
        if image_data is not None:
            return normalize_and_convert_to_png(image_data)
    except Exception:
        pass 
    
    # H5PY Fallback
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mat') as tmp:
            tmp.write(mat_bytes)
            tmp_path = tmp.name
        with h5py.File(tmp_path, 'r') as f:
            image_data = None
            if 'cjdata' in f and 'image' in f['cjdata']:
                data = f['cjdata']['image'][()]
                image_data = np.array(data).T
            if image_data is None:
                for key in ['image', 'img', 'data']:
                    if key in f:
                        image_data = np.array(f[key][()]).T
                        break
            if image_data is not None:
                return normalize_and_convert_to_png(image_data)
    except Exception as e:
        raise ValueError(f"Failed to process v7.3 MAT file: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass 
    raise ValueError("Could not extract image data from .mat file")

# ============================================================================
# ANALYTICS LOGIC (Grad-CAM & Segmentation)
# ============================================================================

def generate_heatmap(model, input_tensor, target_class_idx):
    """
    Generates Grad-CAM heatmap for EfficientNet B0
    """
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Hook into the last convolutional layer of EfficientNet B0 (timm)
    # usually 'conv_head' or 'blocks[-1]'
    target_layer = model.conv_head
    
    # Register hooks
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    # Forward
    model.zero_grad()
    output = model(input_tensor)
    
    # Backward
    score = output[:, target_class_idx]
    score.backward()

    # Clean hooks
    handle_f.remove()
    handle_b.remove()

    if not gradients or not activations:
        return None

    # Grad-CAM calculation
    grads = gradients[0].cpu().data.numpy()[0] # (Channels, H, W)
    fmaps = activations[0].cpu().data.numpy()[0] # (Channels, H, W)
    
    weights = np.mean(grads, axis=(1, 2)) # Global Average Pooling on Gradients
    cam = np.zeros(fmaps.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmaps[i, :, :]

    cam = np.maximum(cam, 0) # ReLU
    
    # Normalize
    if np.max(cam) > 0:
        cam = cam / np.max(cam)
    else:
        cam = cam # All zeros

    return cam

def process_image(image_bytes, rotation_angle=0):
    img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    if rotation_angle != 0:
        img_pil = img_pil.rotate(rotation_angle, expand=True)
    img_gray = img_pil.convert('L')
    
    results = {}
    
    # 1. Classification
    detected_class_idx = 2 # Default No Tumor
    
    if models_registry['classifier']:
        # IMPORTANT: Enable grad for Grad-CAM
        input_tensor = trans_tumor(img_pil).unsqueeze(0).to(Config.DEVICE)
        
        # We need gradients for Grad-CAM, even in inference
        with torch.set_grad_enabled(True):
            outputs = models_registry['classifier'](input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            detected_class_idx = pred_idx.item()
            
            all_probs = probs[0].detach().cpu().numpy()
            results['tumor_class'] = Config.TUMOR_CLASSES[detected_class_idx]
            results['tumor_conf'] = float(conf.item())
            results['tumor_probs'] = {Config.TUMOR_CLASSES[i]: float(all_probs[i]) for i in range(4)}

            # --- OPTION 2: GRAD-CAM ---
            if results['tumor_class'] != 'No Tumor':
                cam = generate_heatmap(models_registry['classifier'], input_tensor, detected_class_idx)
                if cam is not None:
                    # Resize Heatmap to original image size
                    cam_img = Image.fromarray(np.uint8(cam * 255), 'L')
                    cam_img = cam_img.resize(img_pil.size, Image.BILINEAR)
                    cam_arr = np.array(cam_img) / 255.0
                    
                    # Apply Colormap (Jet)
                    # cm.jet returns RGBA 0-1 float
                    heatmap_colored = cm.jet(cam_arr) 
                    heatmap_colored = np.uint8(heatmap_colored * 255)
                    
                    # Create RGBA Image
                    # Set Alpha channel based on intensity (so low activation is transparent)
                    heatmap_pil = Image.fromarray(heatmap_colored)
                    r, g, b, a = heatmap_pil.split()
                    
                    # Make regions with 0 activation transparent
                    # Use the cam_img itself as a mask for alpha
                    final_heatmap = Image.merge('RGBA', (r, g, b, cam_img))
                    
                    buff_cam = io.BytesIO()
                    final_heatmap.save(buff_cam, format="PNG")
                    results['heatmap_image'] = base64.b64encode(buff_cam.getvalue()).decode('utf-8')
    
    # 2. Planar Detection
    detected_plane = 'axial'
    if models_registry['planar']:
        with torch.no_grad():
            input_tensor = trans_gray_planar(img_gray).unsqueeze(0).to(Config.DEVICE)
            outputs = models_registry['planar'](input_tensor)
            _, pred_idx = torch.max(outputs, 1)
            detected_plane = Config.PLANAR_CLASSES[pred_idx.item()]
            results['planar_class'] = detected_plane
    
    # 3. Segmentation (Option 1 Impl)
    seg_key = f'seg_{detected_plane}'
    
    if results['tumor_class'] != 'No Tumor' and models_registry.get(seg_key):
            with torch.no_grad():
                input_tensor = trans_gray_seg(img_gray).unsqueeze(0).to(Config.DEVICE)
                output = models_registry[seg_key](input_tensor)
                probs = torch.softmax(output, dim=1)
                pred_mask = probs[0, 1, :, :].cpu().numpy()
                
                # Resize mask to original
                binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
                mask_img = Image.fromarray(binary_mask, mode='L')
                mask_img = mask_img.resize(img_pil.size, Image.BILINEAR)
                
                # --- OPTION 1: RETURN TRANSPARENT MASK LAYER ---
                # Create a Solid Red Image
                red = Image.new("L", img_pil.size, 255)
                green = Image.new("L", img_pil.size, 50)
                blue = Image.new("L", img_pil.size, 50)
                
                # Use the binary mask as the Alpha channel
                # Where mask is white (tumor), image is Red. Where mask is black, image is transparent.
                overlay = Image.merge("RGBA", (red, green, blue, mask_img))
                
                buff = io.BytesIO()
                overlay.save(buff, format="PNG")
                results['segmentation_image'] = base64.b64encode(buff.getvalue()).decode('utf-8')
            
    # Return Original
    buff_orig = io.BytesIO()
    img_pil.save(buff_orig, format="PNG")
    results['original_image'] = base64.b64encode(buff_orig.getvalue()).decode('utf-8')
    
    return results

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    rotation = int(request.form.get('rotation', 0))

    try:
        img_bytes = file.read()
        fname = file.filename.lower()
        
        if fname.endswith('.dcm'):
            img_bytes = convert_dicom_to_png_bytes(img_bytes)
        elif fname.endswith('.mat'):
            img_bytes = convert_mat_to_png_bytes(img_bytes)
            
        results = process_image(img_bytes, rotation_angle=rotation)
        return jsonify(results)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable (HF Spaces uses 7860)
    port = int(os.environ.get('PORT', 7860))
    # Disable debug in production
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)