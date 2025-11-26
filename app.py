import os
import io
import base64
import tempfile
import numpy as np
from PIL import Image

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

# Configuration class for paths, constants, and device setup
class Config:
    # Paths to model files
    MODEL_TUMOR_CLASS = 'models/efficientnet_b0_best.pth'
    MODEL_PLANAR = 'models/plane_classifier.pth'
    
    # Mapping of planar output to segmentation model paths
    MODEL_SEGMENTATION = {
        'axial': 'models/ax_best_model.pth',
        'coronal': 'models/co_best_model.pth',
        'sagittal': 'models/sa_best_model.pth'
    }

    # Class labels for tumor and planar classification
    TUMOR_CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    PLANAR_CLASSES = ['axial', 'coronal', 'sagittal'] 
    
    # Image sizes for classification and segmentation
    IMG_SIZE_CLS = 224
    IMG_SIZE_SEG = 512
    
    # Device configuration (GPU if available, otherwise CPU)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================

# PlaneClassifier: A model for classifying image planes (axial, coronal, sagittal)
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

# Atrous Spatial Pyramid Pooling (ASPP) module for segmentation
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

# DeepLabHead: Decoder head for DeepLabV3+ segmentation model
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

# DeepLabV3Plus: Full segmentation model with ResNet backbone and DeepLabHead
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

# Flask app initialization and configuration
app = Flask(__name__)
app.secret_key = 'bme-mri-project-secret'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global registry for loaded models
models_registry = {
    'classifier': None,
    'planar': None,
    'seg_axial': None,
    'seg_coronal': None,
    'seg_sagittal': None
}

# Function to load models into memory
def load_models():
    print("⏳ Loading Models into Memory...")
    
    if os.path.exists(Config.MODEL_TUMOR_CLASS):
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

# Preprocessing transforms for different tasks
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

# Helper function to normalize and convert 2D array to PNG bytes
def normalize_and_convert_to_png(data):
    data = data.astype(float)
    data = (np.maximum(data, 0) / data.max()) * 255.0
    data = np.uint8(data)
    
    img = Image.fromarray(data)
    buff = io.BytesIO()
    img.save(buff, format="PNG")
    return buff.getvalue()

# Helper function to process DICOM files
def convert_dicom_to_png_bytes(dicom_bytes):
    try:
        ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
        return normalize_and_convert_to_png(ds.pixel_array)
    except Exception as e:
        raise ValueError(f"Failed to process DICOM: {str(e)}")

# Helper function to process MATLAB .mat files
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

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mat') as tmp:
            tmp.write(mat_bytes)
            tmp_path = tmp.name
        
        with h5py.File(tmp_path, 'r') as f:
            image_data = None
            
            if 'cjdata' in f:
                if 'image' in f['cjdata']:
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
            try:
                os.remove(tmp_path)
            except Exception:
                pass 

    raise ValueError("Could not extract image data from .mat file (tried Scipy and HDF5 methods)")

# ============================================================================

# Function to process an image and perform classification and segmentation
def process_image(image_bytes, rotation_angle=0):
    img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    if rotation_angle != 0:
        img_pil = img_pil.rotate(rotation_angle, expand=True)
    
    img_gray = img_pil.convert('L')
    
    results = {}
    
    if models_registry['classifier']:
        input_tensor = trans_tumor(img_pil).unsqueeze(0).to(Config.DEVICE)
        with torch.no_grad():
            outputs = models_registry['classifier'](input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            
            all_probs = probs[0].cpu().numpy()
            results['tumor_class'] = Config.TUMOR_CLASSES[pred_idx.item()]
            results['tumor_conf'] = float(conf.item())
            results['tumor_probs'] = {
                Config.TUMOR_CLASSES[i]: float(all_probs[i]) 
                for i in range(4)
            }
    
    detected_plane = 'axial'
    if models_registry['planar']:
        input_tensor = trans_gray_planar(img_gray).unsqueeze(0).to(Config.DEVICE)
        with torch.no_grad():
            outputs = models_registry['planar'](input_tensor)
            _, pred_idx = torch.max(outputs, 1)
            detected_plane = Config.PLANAR_CLASSES[pred_idx.item()]
            results['planar_class'] = detected_plane
    
    seg_key = f'seg_{detected_plane}'
    mask_b64 = None
    
    if results['tumor_class'] != 'No Tumor' and models_registry.get(seg_key):
            input_tensor = trans_gray_seg(img_gray).unsqueeze(0).to(Config.DEVICE)
            with torch.no_grad():
                output = models_registry[seg_key](input_tensor)
                probs = torch.softmax(output, dim=1)
                pred_mask = probs[0, 1, :, :].cpu().numpy()
                
                binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
                
                mask_img = Image.fromarray(binary_mask, mode='L')
                mask_img = mask_img.resize(img_pil.size, Image.BILINEAR)
                
                overlay = Image.new("RGBA", img_pil.size, (255, 0, 0, 0))
                mask_arr = np.array(mask_img)
                overlay_arr = np.array(overlay)
                overlay_arr[mask_arr > 127] = [255, 50, 50, 100] 
                
                final_overlay = Image.fromarray(overlay_arr, 'RGBA')
                combined = Image.alpha_composite(img_pil.convert('RGBA'), final_overlay)
                
                buff = io.BytesIO()
                combined.save(buff, format="PNG")
                mask_b64 = base64.b64encode(buff.getvalue()).decode('utf-8')
            
    results['segmentation_image'] = mask_b64
    
    buff_orig = io.BytesIO()
    img_pil.save(buff_orig, format="PNG")
    results['original_image'] = base64.b64encode(buff_orig.getvalue()).decode('utf-8')
    
    return results

# ============================================================================

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling predictions
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

# Entry point for running the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
