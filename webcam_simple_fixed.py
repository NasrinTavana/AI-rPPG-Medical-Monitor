# webcam_simple.py - FIXED VERSION
import json
from collections import deque

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, welch, detrend

print("=" * 50)
print("🚀 ME-rPPG - دوربین آنلاین + ضربان قلب")
print("=" * 50)

# ==========================================================
# توابع مورد نیاز
# ==========================================================

def load_state(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_model(path):
    import onnx
    import onnxruntime as ort
    
    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    
    model_proto = onnx.load(path)
    
    base_state = {}
    for init in model_proto.graph.initializer:
        try:
            arr = onnx.numpy_helper.to_array(init)
            if init.name in input_names and init.name != 'arg_0.1':
                base_state[init.name] = arr.copy()
        except:
            pass
    
    for inp in session.get_inputs():
        if inp.name != 'arg_0.1' and inp.name not in base_state:
            shape = [1 if s is None else s for s in inp.shape]
            base_state[inp.name] = np.zeros(shape, dtype=np.float32)
    
    def run(img, state):
        img_in = img[None, None].astype(np.float32)
        inputs = {'arg_0.1': img_in}
        
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                inputs[k] = v.astype(np.float32)
            else:
                inputs[k] = np.array(v, dtype=np.float32)
        
        for k, v in base_state.items():
            if k not in inputs:
                inputs[k] = v.astype(np.float32) if isinstance(v, np.ndarray) else np.array(v, dtype=np.float32)
        
        results = session.run(output_names, inputs)
        bvp = float(results[0].flat[0]) if len(results) > 0 else 0.0
        
        new_state = {}
        for i, name in enumerate(output_names[1:], 1):
            if i < len(results):
                new_state[name] = results[i]
        
        if not new_state:
            new_state = state.copy()
        
        return bvp, new_state
    
    run.base_state = base_state
    return run

def bandpass(signal, fs):
    low, high = 0.7, 3.5
    nyquist = fs / 2
    if nyquist <= high:
        return signal
    b, a = butter(3, [low / nyquist, high / nyquist], btype="band")
    return filtfilt(b, a, signal)

def get_hr(signal, sr=30, hr_min=40, hr_max=180):
    signal = np.asarray(signal).flatten()
    if len(signal) < 30:
        return 0

    signal = detrend(signal) - np.mean(signal)
    std = np.std(signal)
    if std > 1e-6:
        signal /= std

    try:
        signal = bandpass(signal, sr)
    except Exception:
        pass

    f, pxx = welch(
        signal, fs=sr,
        window="hann",
        nperseg=min(256, len(signal)),
        noverlap=min(128, len(signal) // 2),
        scaling="density",
    )

    mask = (f >= hr_min / 60) & (f <= hr_max / 60)
    if np.sum(mask) == 0:
        return 0

    hr = f[mask][np.argmax(pxx[mask])] * 60
    return float(hr) if hr_min <= hr <= hr_max else 0

# ==========================================================
# بارگذاری مدل
# ==========================================================

try:
    state = load_state('state.json')
    model = load_model('model.onnx')
    print("✅ مدل بارگذاری شد")
except Exception as e:
    print(f"❌ خطا: {e}")
    exit()

# ==========================================================
# باز کردن وب‌کم - با 3 روش مختلف
# ==========================================================

cap = None
backends = [
    (cv2.CAP_DSHOW, "DirectShow"),
    (cv2.CAP_MSMF, "MSMF"),
    (cv2.CAP_V4L2, "V4L2"),
]

for backend, name in backends:
    print(f"⏳ تلاش با {name}...")
    cap = cv2.VideoCapture(0, backend)
    if cap.isOpened():
        print(f"✅ وب‌کم با {name} باز شد!")
        break
    cap = None

if cap is None:
    # روش آخر: بدون backend مشخص
    print("⏳ تلاش با backend پیش‌فرض...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ وب‌کم باز نشد!")
        print("💡 راه‌حل‌ها:")
        print("   1. دوربین را بررسی کنید")
        print("   2. برنامه دیگری که از دوربین استفاده می‌کند را ببندید")
        print("   3. کامپیوتر را ریستارت کنید")
        exit()

print("📌 برای خروج 'q' را فشار دهید")

# ==========================================================
# اجرا
# ==========================================================

bvp_buffer = deque(maxlen=150)
hr_value = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ خطا در خواندن فریم!")
        break
    
    frame_count += 1
    
    # برش از مرکز
    h, w = frame.shape[:2]
    center_x, center_y = w//2, h//2
    crop_size = min(h, w) // 2
    
    x1 = max(0, center_x - crop_size)
    y1 = max(0, center_y - crop_size)
    x2 = min(w, center_x + crop_size)
    y2 = min(h, center_y + crop_size)
    
    face_found = False
    
    try:
        face_roi = frame[y1:y2, x1:x2]
        
        if face_roi.size > 0:
            facial_img = cv2.resize(face_roi, (36, 36), interpolation=cv2.INTER_AREA)
            facial_img = facial_img.astype('float32') / 255.0
            
            output, state = model(facial_img, state)
            bvp_buffer.append(float(output))
            
            if len(bvp_buffer) >= 30 and frame_count % 30 == 0:
                try:
                    hr = get_hr(np.array(list(bvp_buffer)))
                    if 40 < hr < 200:
                        hr_value = hr
                        print(f"💓 ضربان قلب: {hr_value:.1f} BPM | نمونه‌ها: {len(bvp_buffer)}")
                except Exception as e:
                    pass
            
            face_found = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    except Exception as e:
        pass
    
    # نمایش اطلاعات
    if hr_value > 0:
        text = f"❤️ Heart Rate: {hr_value:.1f} BPM"
        color = (0, 255, 0)
    elif face_found:
        text = "⏳ Detecting..."
        color = (0, 165, 255)
    else:
        text = "😕 No Face Detected"
        color = (0, 0, 255)
    
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.putText(frame, f"BVP: {len(bvp_buffer)}", (frame.shape[1]-120, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imshow('ME-rPPG - Heart Rate Monitor', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("📊 خلاصه:")
print(f"   - فریم‌های پردازش شده: {frame_count}")
print(f"   - نمونه‌های BVP: {len(bvp_buffer)}")
if hr_value > 0:
    print(f"   - ✅ ضربان قلب: {hr_value:.1f} BPM")
else:
    print("   - ❌ ضربان قلبی ثبت نشد")
print("=" * 50)