import io
import time
import torch
import requests
import numpy as np
import cv2
import face_recognition
from PIL import Image

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse


app = FastAPI()

# ==========================================================
# ✅ HEALTH CHECK ENDPOINT
# ==========================================================
@app.get("/")
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Face Verification Server Running", "port": 5001}

# ==========================================================
# ✅ YOLO SPOOF CLASSIFIER (For Real vs Fake Face Detection)
# ==========================================================
class SimpleSpoofDetector:
    def __init__(self, model_path="best.pt", device=None):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ YOLO model loaded on: {self.device}")

    def predict(self, image_array):
        """Run YOLO classification and return probabilities"""
        results = self.model.predict(image_array, imgsz=640, device=self.device, verbose=False)
        probs = results[0].probs
        if probs is None:
            return None
        prob_list = probs.data.cpu().numpy()
        return prob_list


# ==========================================================
# ✅ IMAGE COMPRESSION (Only for Face Verification)
# ==========================================================
def compress_and_resize_image(image_bytes, max_size=800):
    """Compress & resize large images to improve face detection"""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    scale = min(max_size / width, max_size / height)

    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        print(f"🧩 Compressed image to: {new_width}x{new_height}")
    else:
        print(f"✅ Image already small enough ({width}x{height})")

    compressed_io = io.BytesIO()
    img.save(compressed_io, format="JPEG", quality=70, optimize=True)
    compressed_io.seek(0)

    # Convert to numpy array for OpenCV usage
    img_array = np.array(Image.open(compressed_io))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_bgr


# ==========================================================
# ✅ FACE VERIFICATION FUNCTION
# ==========================================================
def verify_face_simple(test_image_bytes, reference_image_url):
    """Compare employee face with captured image"""

    print("\n👤 === FACE VERIFICATION STARTING ===")

    # Step 1: Compress live/test image before face detection
    print("🔧 Compressing test image for better face detection...")
    test_img_array = compress_and_resize_image(test_image_bytes)
    test_img = Image.fromarray(cv2.cvtColor(test_img_array, cv2.COLOR_BGR2RGB))

    # Step 2: Load reference image from HRM server
    print("🌐 Downloading employee reference image...")
    ref_resp = requests.get(reference_image_url)
    ref_img = Image.open(io.BytesIO(ref_resp.content))
    if ref_img.mode != "RGB":
        ref_img = ref_img.convert("RGB")

    # Convert to numpy
    ref_array = np.array(ref_img)
    test_array = np.array(test_img)
    print(f"📷 Reference image size: {ref_array.shape}")
    print(f"📷 Test image size: {test_array.shape}")

    # Step 3: Detect faces
    print("🔍 Detecting faces (using HOG model)...")
    ref_faces = face_recognition.face_locations(ref_array, model="hog")
    test_faces = face_recognition.face_locations(test_array, model="hog")

    # Fallback: If no faces found, try CNN
    if len(test_faces) == 0:
        print("⚠️ No face found with HOG, trying CNN model...")
        test_faces = face_recognition.face_locations(test_array, model="cnn")

    if len(ref_faces) == 0 or len(test_faces) == 0:
        print("❌ Face not found in one or both images.")
        return {"status": False, "message": "Face not found in image"}

    # Step 4: Face encodings
    print("🔢 Encoding faces...")
    ref_enc = face_recognition.face_encodings(ref_array, ref_faces)
    test_enc = face_recognition.face_encodings(test_array, test_faces)

    if not ref_enc or not test_enc:
        print("❌ Encoding failed.")
        return {"status": False, "message": "Encoding failed"}

    # Step 5: Compare faces
    print("🧠 Comparing faces...")
    result = face_recognition.compare_faces([ref_enc[0]], test_enc[0])[0]
    distance = face_recognition.face_distance([ref_enc[0]], test_enc[0])[0]

    print(f"🎯 Match: {result}, Distance: {distance:.4f}")

    return {
        "status": bool(result),
        "distance": float(distance),
        "message": "Face matched" if result else "Face not matched",
    }


# ==========================================================
# ✅ API ENDPOINT FOR VERIFICATION
# ==========================================================
@app.post("/verify")
async def verify_employee(
    employee_id: str = Form(...),
    employee_name: str = Form(...),
    department: str = Form(...),
    reference_image_url: str = Form(...),
    captured_image: UploadFile = Form(...),
):
    print(f"\n=== 🧾 COMPLETE VERIFICATION FOR EMPLOYEE {employee_id} ===")

    start_time = time.time()
    image_bytes = await captured_image.read()

    # --------------------------
    # Step 1: YOLO Spoof Detection
    # --------------------------
    print("\n🔍 === SPOOF DETECTION STARTING ===")
    img_np = np.frombuffer(image_bytes, np.uint8)
    full_img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    detector = SimpleSpoofDetector("best.pt")
    probs = detector.predict(full_img_bgr)

    if probs is None:
        return JSONResponse({
            "status": False,
            "message": "YOLO failed to classify image"
        })

    real_prob = float(probs[0])
    spoof_prob = float(probs[1])
    print(f"📊 Raw probabilities: [Real={real_prob:.5f}, Spoof={spoof_prob:.5f}]")

    if real_prob < 0.7:
        print("❌ Spoof detected, stopping verification.")
        return JSONResponse({
            "status": False,
            "message": "Spoof detected",
            "real_prob": real_prob,
            "spoof_prob": spoof_prob
        })

    print("✅ REAL IMAGE detected. Proceeding to face verification...")

    # --------------------------
    # Step 2: Face Verification
    # --------------------------
    verification = verify_face_simple(image_bytes, reference_image_url)

    total_time = time.time() - start_time
    print(f"⏱️ Total processing time: {total_time:.2f} sec")

    return JSONResponse({
        "employee_id": employee_id,
        "employee_name": employee_name,
        "department": department,
        "spoof_check": {
            "real_prob": real_prob,
            "spoof_prob": spoof_prob
        },
        "verification": verification,
        "processing_time": round(total_time, 2),
    })


# ==========================================================
# ✅ SERVE HTML UI
# ==========================================================
@app.get("/ui")
async def serve_ui():
    from fastapi.responses import HTMLResponse
    try:
        with open("test_main_ui.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>UI file not found. Please ensure test_main_ui.html is in the same directory.</h1>")


# ==========================================================
# ✅ MAIN ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    
    def open_browser():
        import time
        time.sleep(2)  # Wait for server to start
        webbrowser.open('http://localhost:5001/ui')
    
    print("🚀 Face Verification Server Starting on port 5001...")
    print("🌐 Opening browser UI automatically...")
    print("📡 Server: http://localhost:5001")
    print("🖥️  UI: http://localhost:5001/ui")
    print("📋 API Docs: http://localhost:5001/docs")
    
    # Open browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    uvicorn.run(app, host="0.0.0.0", port=5001)
    uvicorn.run(app, host="0.0.0.0", port=5001)
