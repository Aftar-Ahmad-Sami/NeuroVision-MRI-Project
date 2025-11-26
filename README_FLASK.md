# Brain Tumor Classification Web Application

## 🚀 Quick Start Guide

This is a complete Flask web application for brain tumor classification using your trained EfficientNet-B0 model.

## 📁 Project Structure

```
your_project/
├── app.py                          # Main Flask application
├── templates/
│   └── index.html                  # Web GUI
├── uploads/                        # Uploaded images (created automatically)
├── efficientnet_b0_best.pth       # YOUR MODEL FILE (place here)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔧 Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**OR install manually:**

```bash
pip install Flask==3.0.0 torch==2.1.0 torchvision==0.16.0 timm==0.9.12 Pillow==10.1.0
```

### Step 2: Place Your Model File

Copy your trained model file to the project directory:

```bash
# Your model file should be named:
efficientnet_b0_best.pth
```

**IMPORTANT:** The file must be in the same directory as `app.py`

## ▶️ Running the Application

### Start the server:

```bash
python app.py
```

You should see:

```
======================================================================
BRAIN TUMOR CLASSIFICATION API
======================================================================
Loading model from: efficientnet_b0_best.pth
Device: cuda  # or cpu
✅ Model loaded successfully!
Model parameters: 4,012,672
======================================================================

======================================================================
Starting Flask Server...
======================================================================
Access the web interface at: http://127.0.0.1:5000
API endpoint: http://127.0.0.1:5000/api/predict
Press CTRL+C to stop the server
======================================================================
```

## 🌐 Access the Application

### Web Interface (GUI):
Open your browser and go to:
```
http://127.0.0.1:5000
```

### API Endpoint:
```
http://127.0.0.1:5000/api/predict
```

## 📝 How to Use

### Using Web Interface:

1. **Open** `http://127.0.0.1:5000` in your browser
2. **Upload** a brain MRI scan image (PNG, JPG, JPEG)
3. **Click** "Analyze Image" button
4. **View** results with predictions and confidence scores

### Using API (with Python):

```python
import requests

url = 'http://127.0.0.1:5000/api/predict'
files = {'file': open('brain_scan.jpg', 'rb')}
response = requests.post(url, files=files)
result = response.json()

print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

### Using API (with cURL):

```bash
curl -X POST -F "file=@brain_scan.jpg" http://127.0.0.1:5000/api/predict
```

### Using API (with Postman):

1. Method: `POST`
2. URL: `http://127.0.0.1:5000/api/predict`
3. Body: `form-data`
4. Key: `file` (type: File)
5. Value: Select your image file

## 📊 API Response Format

```json
{
  "predicted_class": "Glioma",
  "predicted_index": 0,
  "confidence": 0.9876,
  "probabilities": {
    "Glioma": 0.9876,
    "Meningioma": 0.0054,
    "No Tumor": 0.0032,
    "Pituitary": 0.0038
  },
  "filename": "brain_scan.jpg",
  "timestamp": "20251026_165432"
}
```

## 🔌 Available Endpoints

### 1. **GET** `/` - Web Interface
Returns the HTML GUI page

### 2. **POST** `/predict` - Web Upload Prediction
- Accepts: `multipart/form-data` with `file` field
- Returns: JSON with prediction + base64 image
- Used by web interface

### 3. **POST** `/api/predict` - API Prediction
- Accepts: `multipart/form-data` with `file` field  
- Returns: JSON with prediction only
- For external applications

### 4. **GET** `/health` - Health Check
- Returns: Server status and model info
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "classes": ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
}
```

## 🎨 Features

### Web Interface:
- ✅ Modern, responsive design
- ✅ Drag-and-drop file upload
- ✅ Real-time image preview
- ✅ Animated probability bars
- ✅ Mobile-friendly
- ✅ Error handling with user-friendly messages

### API:
- ✅ RESTful endpoints
- ✅ JSON responses
- ✅ File validation
- ✅ Error handling
- ✅ Health check endpoint

### Model:
- ✅ EfficientNet-B0 (4.01M parameters)
- ✅ 99.80% accuracy
- ✅ GPU/CPU support
- ✅ Fast inference (< 1 second)

## ⚙️ Configuration

Edit `app.py` to customize:

```python
class Config:
    MODEL_PATH = 'efficientnet_b0_best.pth'  # Your model path
    NUM_CLASSES = 4
    CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    IMG_SIZE = 224
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
```

## 🐛 Troubleshooting

### Issue: "Model file not found"
**Solution:** Make sure `efficientnet_b0_best.pth` is in the same directory as `app.py`

### Issue: "CUDA out of memory"
**Solution:** The app will automatically use CPU if GPU is unavailable

### Issue: "Port 5000 already in use"
**Solution:** Change port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

### Issue: "Module not found"
**Solution:** Install missing packages:
```bash
pip install -r requirements.txt
```

## 🌍 Deploying to Production

### For local network access:
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```
Access from other devices: `http://YOUR_IP:5000`

### For production deployment:
Use a production WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 🔐 Security Notes

- Change `app.secret_key` in production
- Use HTTPS in production
- Implement authentication if needed
- Validate all inputs
- Set appropriate CORS headers if needed

## 📱 Testing

### Test with sample images:
```bash
# Upload a test image via API
curl -X POST -F "file=@test_image.jpg" http://127.0.0.1:5000/api/predict
```

### Check server health:
```bash
curl http://127.0.0.1:5000/health
```

## 📊 Performance

- **Model Size:** 4.01M parameters (~16MB file)
- **Inference Time:** < 1 second per image
- **Max File Size:** 16MB
- **Supported Formats:** PNG, JPG, JPEG
- **Concurrent Requests:** Depends on hardware

## 🎯 Next Steps

1. ✅ Test with your own images
2. ✅ Share with colleagues (local network)
3. ✅ Deploy to cloud (AWS, Azure, Heroku)
4. ✅ Add authentication
5. ✅ Add database for results storage
6. ✅ Add batch processing
7. ✅ Add result visualization

## 📄 License

This application is for research and educational purposes.

## ⚠️ Medical Disclaimer

This is an AI-assisted diagnostic tool for research purposes only. Always consult qualified medical professionals for diagnosis and treatment. Do not use for clinical decisions without proper validation and approval.

## 🆘 Support

If you encounter issues:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure model file is in correct location
4. Check file permissions
5. Verify network connectivity

---

**Created for Brain Tumor Classification Research**

Model Accuracy: 99.80% | Classes: 4 | Architecture: EfficientNet-B0
