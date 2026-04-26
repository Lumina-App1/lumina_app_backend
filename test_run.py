"""
FIXED CAMERA TEST - No waiting, camera never freezes
Uses threading so camera runs independently
"""

import cv2
import time
import requests
import base64
import threading
from queue import Queue

BACKEND_URL = "http://localhost:5000/detect"

# Queue for frames to process (so camera doesn't wait)
frame_queue = Queue(maxsize=5)
result_queue = Queue()

# Global variable for latest detection result
latest_result = {"label": "Waiting...", "direction": "", "meters": 0}

def camera_capture():
    """Captures frames continuously - NEVER blocks"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera error!")
        return
    
    # Set camera properties for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # Smaller = faster
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Only process every 5th frame
        if frame_count % 5 == 0:
            # Put frame in queue for processing (non-blocking)
            if frame_queue.qsize() < 5:
                frame_queue.put(frame.copy())
        
        # Always show camera (never freezes)
        # Draw current detection result on frame
        if latest_result["label"] != "Waiting...":
            cv2.putText(frame, f"Object: {latest_result['label']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Direction: {latest_result['direction']}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Distance: {latest_result['meters']}m", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("AI Vision - Camera Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def backend_processor():
    """Processes frames from queue in background - camera never waits"""
    global latest_result
    
    print("🔄 Backend processor started...")
    
    # Warm up YOLO (send a dummy request to load model)
    print("📦 Loading YOLO model (first time takes 5-10 seconds)...")
    dummy_frame = cv2.imread("test.jpeg") if cv2.imread("test.jpeg") is not None else None
    
    while True:
        try:
            # Get frame from queue (wait if empty)
            frame = frame_queue.get(timeout=1)
            
            # Resize for faster processing
            frame = cv2.resize(frame, (320, 240))
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            # Send to backend with timeout
            try:
                start_time = time.time()
                response = requests.post(
                    BACKEND_URL,
                    json={'image': base64_image},
                    timeout=10
                )
                
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('detection_found'):
                        best = data['primary_detection']
                        latest_result = {
                            "label": best['label'],
                            "direction": best['direction'],
                            "meters": best['meters'],
                            "message": best['voice_message']
                        }
                        print(f"\n🎯 {best['label'].upper()} | {best['direction']} | {best['meters']}m | Time: {elapsed:.1f}s")
                        print(f"   💬 {best['voice_message'][:80]}...")
                    else:
                        latest_result = {"label": "No objects", "direction": "", "meters": 0}
                        print(f"\r🔍 No objects... (time: {elapsed:.1f}s)", end="")
                        
            except requests.exceptions.Timeout:
                print(f"\r⏰ Timeout... (processing too slow)", end="")
            except Exception as e:
                print(f"\r❌ Error: {e}", end="")
                
        except:
            pass  # Queue empty, continue

def main():
    print("=" * 60)
    print("📷 CAMERA FIX TEST - No Freeze!")
    print("=" * 60)
    print("\n✅ Make sure: python app.py is RUNNING in another terminal")
    print("📹 Camera will run SMOOTHLY - no freezing!")
    print("🎯 Point camera at objects")
    print("❌ Press 'q' to quit")
    print("=" * 60)
    print("\n🚀 Starting...\n")
    
    # Start backend processor thread
    processor_thread = threading.Thread(target=backend_processor, daemon=True)
    processor_thread.start()
    
    # Run camera in main thread (always smooth)
    camera_capture()

if __name__ == "__main__":
    main()