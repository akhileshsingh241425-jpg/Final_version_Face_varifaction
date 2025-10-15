#!/usr/bin/env python3
"""
🔥 SIMPLE SPOOF DETECTION FOCUSED SERVER 🔥
Step 1: YOLO Spoof Detection FIRST
Step 2: Face Verification ONLY if not spoofed
"""

import os
import io
import time
import requests
import numpy as np
from PIL import Image
import face_recognition
import torch
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Fix for PyTorch 2.9+ weights_only issue
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# API Configuration
EMPLOYEE_API_URL = "https://hrm.umanerp.com/api/users/getEmployee"

class SimpleSpoofDetector:
    def __init__(self):
        """Initialize YOLO spoof detection model"""
        print("🚀 Loading Simple Spoof Detector...")
        
        # YOLO Model Path
        self.model_path = 'memory_optimized_30/yolov8m_1024_30ep_mem/weights/best.pt'
        
        try:
            # Fix for PyTorch 2.9+ - allow loading of ultralytics models
            import torch.serialization
            with torch.serialization.safe_globals(['ultralytics.nn.tasks.ClassificationModel']):
                self.model = YOLO(self.model_path)
            print(f"✅ YOLO Model loaded: {self.model_path}")
            print(f"📊 Model classes: {self.model.names}")
            self.model_loaded = True
        except Exception as e:
            print(f"❌ YOLO Model loading failed: {str(e)}")
            print("🔧 Trying alternative loading method...")
            try:
                # Fallback: Load with weights_only=False (less secure but works)
                os.environ['TORCH_WEIGHTS_ONLY'] = '0'
                self.model = YOLO(self.model_path)
                print(f"✅ YOLO Model loaded (fallback method)")
                self.model_loaded = True
            except Exception as e2:
                print(f"❌ Fallback also failed: {str(e2)}")
                self.model_loaded = False
                self.model = None
        
        print("🎯 Simple Spoof Detector Ready!")
    
    def detect_spoof_simple(self, image_array: np.ndarray):
        """Simple YOLO CLASSIFICATION spoof detection - FIXED FOR CLASSIFICATION MODEL"""
        print(f"\n🔍 === SPOOF DETECTION STARTING ===")
        print(f"📷 Input image shape: {image_array.shape}")
        
        if not self.model_loaded:
            print("❌ Model not loaded - cannot detect spoof")
            return {"spoof_detected": False, "confidence": 0.0, "reason": "model_not_loaded"}
        
        try:
            # Run YOLO CLASSIFICATION (not detection!)
            print(f"🤖 Running YOLO CLASSIFICATION inference...")
            results = self.model(image_array, verbose=False)
            
            print(f"📊 YOLO returned {len(results)} result(s)")
            
            if len(results) > 0:
                result = results[0]
                
                # For CLASSIFICATION model, use 'probs' not 'boxes'
                if hasattr(result, 'probs') and result.probs is not None:
                    probs = result.probs
                    print(f"🎯 Classification probabilities found!")
                    
                    # Get probability values
                    prob_values = probs.data.cpu().numpy()
                    print(f"📊 Raw probabilities: {prob_values}")
                    
                    # Get class probabilities
                    real_confidence = float(prob_values[0])  # Class 0 = 'real'
                    spoof_confidence = float(prob_values[1])  # Class 1 = 'spoof'
                    
                    print(f"📈 Classification Results:")
                    print(f"   Real probability: {real_confidence:.4f} ({real_confidence*100:.1f}%)")
                    print(f"   Spoof probability: {spoof_confidence:.4f} ({spoof_confidence*100:.1f}%)")
                    
                    # Get top prediction
                    predicted_class = probs.top1  # 0 or 1
                    predicted_confidence = probs.top1conf.item()
                    class_name = self.model.names.get(predicted_class, f"class_{predicted_class}")
                    
                    print(f"🎯 Top prediction: {class_name} (confidence: {predicted_confidence:.4f})")
                    
                    # Decision logic: Use spoof confidence threshold
                    spoof_threshold = 0.6  # 60% threshold for spoof detection
                    
                    if spoof_confidence > spoof_threshold:
                        print(f"🚨 SPOOF DETECTED! Spoof confidence: {spoof_confidence:.4f} > threshold {spoof_threshold}")
                        return {
                            "spoof_detected": True,
                            "confidence": spoof_confidence,
                            "confidence_percent": round(spoof_confidence * 100, 1),
                            "reason": "yolo_classification_spoof",
                            "details": f"Spoof probability {spoof_confidence:.3f} > threshold {spoof_threshold}",
                            "real_confidence": round(real_confidence * 100, 1),
                            "spoof_confidence": round(spoof_confidence * 100, 1)
                        }
                    else:
                        print(f"✅ REAL IMAGE detected. Real confidence: {real_confidence:.4f}, Spoof: {spoof_confidence:.4f}")
                        return {
                            "spoof_detected": False,
                            "confidence": real_confidence,
                            "confidence_percent": round(real_confidence * 100, 1),
                            "reason": "yolo_classification_real",
                            "details": f"Real probability {real_confidence:.3f}, Spoof probability {spoof_confidence:.3f}",
                            "real_confidence": round(real_confidence * 100, 1),
                            "spoof_confidence": round(spoof_confidence * 100, 1)
                        }
                else:
                    print("⚠️ No classification probabilities found")
                    return {
                        "spoof_detected": False,
                        "confidence": 0.5,
                        "confidence_percent": 50.0,
                        "reason": "no_probs_assume_real",
                        "details": "No classification probabilities available"
                    }
            else:
                print("⚠️ No YOLO results")
                return {
                    "spoof_detected": False,
                    "confidence": 0.5,
                    "confidence_percent": 50.0,
                    "reason": "no_results_assume_real",
                    "details": "YOLO classification returned no results"
                }
                
        except Exception as e:
            print(f"❌ Spoof detection error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "spoof_detected": False,
                "confidence": 0.0,
                "confidence_percent": 0.0,
                "reason": "error",
                "details": f"Classification error: {str(e)}"
            }
    
    def check_employee_exists(self, employee_id: str):
        """Check if employee exists in HRM system - FIRST validation"""
        try:
            print(f"\n🔍 Checking Employee ID: {employee_id}")
            response = requests.get(f"{EMPLOYEE_API_URL}?empId={employee_id}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('employees'):
                    for emp in data['employees']:
                        if emp.get('employeeId') == employee_id:
                            emp_name = emp.get('fullName', 'Unknown')
                            emp_dept = emp.get('department', 'N/A')
                            print(f"✅ Employee Found: {emp_name} ({emp_dept})")
                            return {
                                "exists": True,
                                "name": emp_name,
                                "department": emp_dept,
                                "message": f"Employee {emp_name} found in system"
                            }
                    
                    print(f"❌ Employee ID {employee_id} not found in database")
                    return {
                        "exists": False,
                        "message": f"Employee ID '{employee_id}' does not exist in HRM system"
                    }
                else:
                    print(f"❌ API returned no employee data")
                    return {
                        "exists": False,
                        "message": "No employee data returned from HRM API"
                    }
            else:
                print(f"❌ API Error: HTTP {response.status_code}")
                return {
                    "exists": False,
                    "message": f"HRM API error: HTTP {response.status_code}"
                }
                
        except Exception as e:
            print(f"❌ Employee check error: {str(e)}")
            return {
                "exists": False,
                "message": f"Employee verification failed: {str(e)}"
            }
    
    def get_employee_photo(self, employee_id: str):
        """Get employee reference photo from HRM API"""
        try:
            print(f"\n👤 Getting employee photo for ID: {employee_id}")
            response = requests.get(f"{EMPLOYEE_API_URL}?empId={employee_id}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('employees'):
                    for emp in data['employees']:
                        if emp.get('employeeId') == employee_id:
                            image_path = emp.get('userImg', '')
                            if image_path:
                                image_url = f"https://hrm.umanerp.com/{image_path}"
                                print(f"✅ Employee found: {emp.get('fullName')}")
                                print(f"🖼️ Image URL: {image_url}")
                                return image_url, emp.get('fullName')
            return None, None
        except Exception as e:
            print(f"❌ Employee API error: {str(e)}")
            return None, None
    
    def verify_face_simple(self, test_image_bytes, employee_id: str):
        """Simple face verification - ONLY if spoof not detected"""
        print(f"\n👤 === FACE VERIFICATION STARTING ===")
        
        # Get reference image
        ref_url, emp_name = self.get_employee_photo(employee_id)
        if not ref_url:
            return {"verified": False, "confidence": 0.0, "message": "Employee not found or no photo"}
        
        try:
            # Download reference image
            ref_response = requests.get(ref_url, timeout=15)
            if ref_response.status_code != 200:
                return {"verified": False, "confidence": 0.0, "message": "Cannot download reference image"}
            
            # Process and optimize images for face recognition
            ref_img = Image.open(io.BytesIO(ref_response.content))
            if ref_img.mode != 'RGB':
                ref_img = ref_img.convert('RGB')
            
            # Optimize reference image
            if ref_img.width > 1280 or ref_img.height > 960:
                ratio = min(1280 / ref_img.width, 960 / ref_img.height)
                new_w, new_h = int(ref_img.width * ratio), int(ref_img.height * ratio)
                ref_img = ref_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                print(f"🔧 Reference image resized to: {new_w}x{new_h}")
            
            ref_array = np.array(ref_img)
            
            test_img = Image.open(io.BytesIO(test_image_bytes))
            if test_img.mode != 'RGB':
                test_img = test_img.convert('RGB')
            
            # Optimize test image
            if test_img.width > 1280 or test_img.height > 960:
                ratio = min(1280 / test_img.width, 960 / test_img.height)
                new_w, new_h = int(test_img.width * ratio), int(test_img.height * ratio)
                test_img = test_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                print(f"🔧 Test image resized to: {new_w}x{new_h}")
            
            test_array = np.array(test_img)
            
            print(f"📷 Reference image: {ref_array.shape}")
            print(f"📷 Test image: {test_array.shape}")
            
            # Get face encodings
            ref_encodings = face_recognition.face_encodings(ref_array)
            test_encodings = face_recognition.face_encodings(test_array)
            
            print(f"👤 Reference faces: {len(ref_encodings)}")
            print(f"👤 Test faces: {len(test_encodings)}")
            
            if len(ref_encodings) == 0:
                return {"verified": False, "confidence": 0.0, "message": "No face in reference image"}
            
            if len(test_encodings) == 0:
                return {"verified": False, "confidence": 0.0, "message": "No face in test image"}
            
            if len(test_encodings) > 1:
                return {"verified": False, "confidence": 0.0, "message": f"Multiple faces detected ({len(test_encodings)})"}
            
            # Compare faces - BALANCED tolerance for better usability
            matches = face_recognition.compare_faces([ref_encodings[0]], test_encodings[0], tolerance=0.5)
            distance = face_recognition.face_distance([ref_encodings[0]], test_encodings[0])[0]
            confidence = max(0, (1 - distance) * 100)
            
            is_match = bool(matches[0])
            
            print(f"🔍 Face comparison (BALANCED tolerance=0.5):")
            print(f"   Match: {is_match}")
            print(f"   Distance: {distance:.4f}")
            print(f"   Confidence: {confidence:.1f}%")
            print(f"   Tolerance used: 0.5 (balanced matching - 50%+ accepted)")
            
            return {
                "verified": is_match,
                "confidence": confidence,
                "message": f"Face match: {is_match} ({confidence:.1f}%)",
                "employee_name": emp_name
            }
            
        except Exception as e:
            print(f"❌ Face verification error: {str(e)}")
            return {"verified": False, "confidence": 0.0, "message": f"Face verification error: {str(e)}"}
    
    def complete_verification(self, employee_id: str, image_bytes):
        """Complete verification: Employee Check FIRST, then Spoof, then Face"""
        start_time = time.time()
        
        print(f"\n🚀 === COMPLETE VERIFICATION FOR EMPLOYEE {employee_id} ===")
        
        # STEP 0: Employee Existence Check - FIRST PRIORITY
        emp_check = self.check_employee_exists(employee_id)
        if not emp_check["exists"]:
            return {
                "success": False,
                "message": f"❌ INVALID EMPLOYEE: {emp_check['message']}",
                "employee_check": emp_check,
                "spoof_detection": {"skipped": "invalid_employee"},
                "face_verification": {"skipped": "invalid_employee"},
                "processing_time": round(time.time() - start_time, 2)
            }
        
        print(f"✅ Employee validation passed: {emp_check['name']}")
        
        # Process and optimize image
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # AUTO-RESIZE for better performance
            original_size = img.size
            print(f"📷 Original image size: {original_size[0]}x{original_size[1]}")
            
            # Max size for optimal performance (based on testing)
            max_width, max_height = 1280, 960
            
            if img.width > max_width or img.height > max_height:
                # Calculate resize ratio to maintain aspect ratio
                ratio = min(max_width / img.width, max_height / img.height)
                new_width = int(img.width * ratio)
                new_height = int(img.height * ratio)
                
                print(f"🔧 Resizing to: {new_width}x{new_height} (ratio: {ratio:.3f})")
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"✅ Image optimized for faster processing")
            else:
                print(f"✅ Image size optimal, no resizing needed")
            
            img_array = np.array(img)
            print(f"📊 Final processing size: {img_array.shape}")
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Image processing error: {str(e)}",
                "processing_time": time.time() - start_time
            }
        
        # STEP 1: SPOOF DETECTION (PRIORITY)
        spoof_result = self.detect_spoof_simple(img_array)
        
        # If SPOOF detected, STOP immediately
        if spoof_result["spoof_detected"]:
            conf_percent = spoof_result["confidence_percent"]
            return {
                "success": False,
                "message": f"🚨 SPOOF DETECTED! Confidence: {conf_percent}% - Access DENIED",
                "spoof_detection": spoof_result,
                "face_verification": {"skipped": "spoof_detected"},
                "processing_time": round(time.time() - start_time, 2)
            }
        
        print(f"✅ Spoof check passed - proceeding to face verification...")
        
        # STEP 2: FACE VERIFICATION (only if not spoofed)
        face_result = self.verify_face_simple(image_bytes, employee_id)
        
        # Final decision
        if face_result["verified"]:
            return {
                "success": True,
                "message": f"✅ VERIFICATION SUCCESSFUL! Employee: {face_result.get('employee_name', 'Unknown')}",
                "spoof_detection": spoof_result,
                "face_verification": face_result,
                "processing_time": round(time.time() - start_time, 2)
            }
        else:
            return {
                "success": False,
                "message": f"❌ FACE VERIFICATION FAILED: {face_result['message']}",
                "spoof_detection": spoof_result,
                "face_verification": face_result,
                "processing_time": round(time.time() - start_time, 2)
            }

# Initialize detector
detector = SimpleSpoofDetector()

# FastAPI app
app = FastAPI(title="🔥 Simple Spoof Detection Server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/verify")
async def verify_employee(
    employee_id: str = Form(...),
    image: UploadFile = File(...)
):
    """Complete verification: Employee Check → Spoof Detection → Face Verification"""
    try:
        # Basic input validation
        if not employee_id or not employee_id.strip():
            return JSONResponse({
                "success": False,
                "message": "❌ Employee ID is required",
                "error_type": "MISSING_EMPLOYEE_ID"
            })
        
        if not image or not image.filename:
            return JSONResponse({
                "success": False,
                "message": "❌ No image file uploaded",
                "error_type": "MISSING_IMAGE"
            })
        
        # Read image
        image_bytes = await image.read()
        if len(image_bytes) == 0:
            return JSONResponse({
                "success": False,
                "message": "❌ Empty image file",
                "error_type": "EMPTY_IMAGE"
            })
        
        # Run complete verification (includes employee check)
        result = detector.complete_verification(employee_id.strip(), image_bytes)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Server error: {str(e)}",
                "processing_time": 0.0
            }
        )

@app.get("/", response_class=HTMLResponse)
async def main_ui():
    """Simple UI for testing"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>🔥 Simple Spoof Detection Test</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; background: #f0f0f0; }
        .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
        input, button { padding: 10px; margin: 10px 0; width: 100%; box-sizing: border-box; }
        button { background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .info { background: #e3f2fd; color: #1976d2; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 Simple Spoof Detection Test</h1>
        <p><strong>Step 1:</strong> YOLO Spoof Detection<br>
           <strong>Step 2:</strong> Face Verification (only if not spoofed)</p>
        
        <form id="testForm" enctype="multipart/form-data">
            <label>Employee ID:</label>
            <input type="text" id="employee_id" name="employee_id" value="4265" required>
            
            <label>Select Image:</label>
            <input type="file" id="image" name="image" accept="image/*" required>
            
            <button type="submit">🚀 Test Verification</button>
        </form>

        <div id="result"></div>
    </div>

    <script>
        document.getElementById('testForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = '<div class="info">🔄 Processing...</div>';
            
            const formData = new FormData();
            formData.append('employee_id', document.getElementById('employee_id').value);
            formData.append('image', document.getElementById('image').files[0]);
            
            try {
                const response = await fetch('/verify', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                let html = `<div class="${result.success ? 'success' : 'error'}">`;
                html += `<h3>${result.message}</h3>`;
                
                if (result.spoof_detection) {
                    const spoof = result.spoof_detection;
                    html += `<p><strong>🛡️ Spoof Detection:</strong><br>`;
                    html += `Detected: ${spoof.spoof_detected ? 'YES' : 'NO'}<br>`;
                    html += `Confidence: ${spoof.confidence_percent}%<br>`;
                    html += `Reason: ${spoof.reason}</p>`;
                }
                
                if (result.face_verification && !result.face_verification.skipped) {
                    const face = result.face_verification;
                    html += `<p><strong>👤 Face Verification:</strong><br>`;
                    html += `Verified: ${face.verified ? 'YES' : 'NO'}<br>`;
                    html += `Confidence: ${face.confidence.toFixed(1)}%<br>`;
                    html += `Message: ${face.message}</p>`;
                }
                
                html += `<p><strong>⏱️ Processing Time:</strong> ${result.processing_time}s</p>`;
                html += '</div>';
                
                resultDiv.innerHTML = html;
                
            } catch (err) {
                resultDiv.innerHTML = `<div class="error">❌ Error: ${err.message}</div>`;
            }
        });
    </script>
</body>
</html>
    """

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "spoof_model_loaded": detector.model_loaded,
        "model_path": detector.model_path
    }

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🔥 SIMPLE SPOOF DETECTION SERVER")
    print("="*50)
    print("🎯 Focus: YOLO Spoof Detection FIRST")
    print("👤 Then: Face Verification (if not spoofed)")
    print(f"🌐 Access: http://localhost:8000")
    print(f"🔍 Health: http://localhost:8000/health")
    print("="*50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")