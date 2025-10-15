# Face Verification System with Spoof Detection

A comprehensive face verification system with anti-spoofing capabilities, employee validation, and optimized image processing.

## Features

- **3-Step Verification Process**:
  1. Employee ID validation against HRM API
  2. Spoof detection using YOLOv8 classification model
  3. Face matching with face_recognition library

- **Optimized Image Processing**: Automatically handles images of any size with intelligent resizing
- **Real-time Detection**: Fast processing with memory-optimized YOLO model
- **Mobile & Desktop UI**: Responsive interfaces for both platforms
- **RESTful API**: FastAPI-based server with CORS support

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for faster inference)
- Webcam/Camera (for live capture)

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/akhileshsingh241425-jpg/Final_version_Face_varifaction.git
cd Final_version_Face_varifaction
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Model Setup**:
The YOLO model is already included in `memory_optimized_30/yolov8m_1024_30ep_mem/weights/best.pt`

## Usage

### Starting the Server

```bash
python simple_spoof_server.py
```

Server will start at `http://localhost:8000`

### API Endpoints

#### POST /verify
Performs complete face verification with spoof detection.

**Request**:
- Content-Type: `multipart/form-data`
- Parameters:
  - `employee_id`: Employee ID (string)
  - `image`: Image file (JPG/PNG)

**Response**:
```json
{
  "success": true,
  "message": "Verification successful",
  "employee_check": {
    "exists": true,
    "employee_id": "12345",
    "name": "John Doe",
    "department": "Engineering"
  },
  "spoof_detection": {
    "is_real": true,
    "confidence": 95.5,
    "label": "real"
  },
  "face_verification": {
    "match": true,
    "confidence": 87.3
  },
  "processing_time": 0.85
}
```

### Using the UI

1. **Desktop Interface**: Open `beautiful_spoof_ui.html` in a browser
2. **Mobile Interface**: Open `mobile_spoof_ui.html` on a mobile device

Both UIs connect to `http://localhost:8000` by default.

## Configuration

Key parameters in `simple_spoof_server.py`:

- **Face Recognition Tolerance**: `0.5` (line 227) - Lower = stricter matching
- **Spoof Detection Threshold**: `60%` (0.6) - Minimum confidence for real face
- **Image Resize Limit**: `1280x960` - Maximum dimensions for optimization
- **Model Path**: `memory_optimized_30/yolov8m_1024_30ep_mem/weights/best.pt`

## API Integration

### Employee Validation API
- Endpoint: `https://hrm.umanerp.com/api/users/getEmployee?empId={id}`
- Used for validating employee existence before processing

## Technical Details

- **YOLO Model**: YOLOv8m classification model (30 epochs, memory-optimized)
- **Classes**: 0 = Real, 1 = Spoof
- **Face Recognition**: dlib-based face encoding with euclidean distance matching
- **Image Processing**: PIL/Pillow with LANCZOS resampling for quality preservation

## Troubleshooting

**Issue**: Face verification failing with valid faces
- **Solution**: Adjust tolerance in line 227 (increase for more lenient matching)

**Issue**: Slow processing with large images
- **Solution**: Images are automatically resized to 1280x960 max

**Issue**: Employee not found
- **Solution**: Verify employee_id exists in HRM system first

## Project Structure

```
.
├── simple_spoof_server.py          # Main server application
├── beautiful_spoof_ui.html         # Desktop UI
├── mobile_spoof_ui.html            # Mobile UI
├── requirements.txt                # Python dependencies
├── memory_optimized_30/            # YOLO model directory
│   └── yolov8m_1024_30ep_mem/
│       └── weights/
│           └── best.pt             # Trained model weights
├── 0/                              # Employee 0 images
└── 1/                              # Employee 1 images
```

## License

MIT License

## Author

Akhilesh Singh

## Support

For issues and questions, please open an issue on GitHub.
