import json
import numpy as np
import onnxruntime as ort
import cv2
import mediapipe as mp
from scipy.signal import butter, filtfilt, welch, detrend

# ==========================================================
# MediaPipe FaceMesh (برای crop_face)
# ==========================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

FOREHEAD = [10, 67, 103, 109, 338, 297, 332]
LEFT_CHEEK = [117, 118, 119, 100, 126]
RIGHT_CHEEK = [346, 347, 348, 329, 355]


# ==========================================================
# Load State
# ==========================================================

def load_state(path):
    """بارگذاری state از فایل JSON"""
    with open(path, 'r') as f:
        return json.load(f)


# ==========================================================
# Load Model
# ==========================================================

def load_model(path):
    """بارگذاری مدل ONNX و بازگرداندن تابع استنتاج"""
    import onnx
    
    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    
    print(f"🔍 Model inputs: {len(input_names)}")
    print(f"🔍 Model outputs: {len(output_names)}")
    
    # پیدا کردن state ها از initializer
    model_proto = onnx.load(path)
    
    base_state = {}
    for init in model_proto.graph.initializer:
        try:
            arr = onnx.numpy_helper.to_array(init)
            if init.name in input_names and init.name != 'arg_0.1':
                base_state[init.name] = arr.copy()
        except:
            pass
    
    # برای state های گمشده
    for inp in session.get_inputs():
        if inp.name != 'arg_0.1' and inp.name not in base_state:
            shape = [1 if s is None else s for s in inp.shape]
            base_state[inp.name] = np.zeros(shape, dtype=np.float32)
    
    print(f" States: {len(base_state)}")
    
    def run(img, state):
        # img: (36, 36, 3) -> (1, 1, 36, 36, 3)
        img_in = img[None, None].astype(np.float32)
        
        inputs = {'arg_0.1': img_in}
        
        # اضافه کردن state ها
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                inputs[k] = v.astype(np.float32)
            else:
                inputs[k] = np.array(v, dtype=np.float32)
        
        # اضافه کردن base_state برای missing keys
        for k, v in base_state.items():
            if k not in inputs:
                inputs[k] = v.astype(np.float32) if isinstance(v, np.ndarray) else np.array(v, dtype=np.float32)
        
        results = session.run(output_names, inputs)
        
        bvp = float(results[0].flat[0]) if len(results) > 0 else 0.0
        
        # به‌روزرسانی state
        new_state = {}
        for i, name in enumerate(output_names[1:], 1):
            if i < len(results):
                new_state[name] = results[i]
        
        if not new_state:
            new_state = state.copy()
        
        return bvp, new_state
    
    run.base_state = base_state
    return run


# ==========================================================
# Crop Face (برای server.py)
# ==========================================================

def crop_face(frame):
    """کراپ صورت با MediaPipe"""
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks is None:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        pts = []

        for idx in FOREHEAD + LEFT_CHEEK + RIGHT_CHEEK:
            lm = landmarks[idx]
            pts.append((int(lm.x * w), int(lm.y * h)))

        pts = np.array(pts)
        x1 = max(0, pts[:, 0].min() - 10)
        y1 = max(0, pts[:, 1].min() - 10)
        x2 = min(w, pts[:, 0].max() + 10)
        y2 = min(h, pts[:, 1].max() + 10)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        roi = cv2.resize(roi, (36, 36), interpolation=cv2.INTER_AREA)
        roi = roi.astype(np.float32) / 255.0
        return roi
    except Exception as e:
        return None


# ==========================================================
# Signal Filters
# ==========================================================

def bandpass(signal, fs):
    low, high = 0.7, 3.5
    nyquist = fs / 2
    if nyquist <= high:
        return signal
    b, a = butter(3, [low / nyquist, high / nyquist], btype="band")
    return filtfilt(b, a, signal)


def signal_quality(sig):
    if len(sig) < 30:
        return 0
    sig = np.array(sig)
    std = np.std(sig)
    if std < 1e-5:
        return 0
    energy = np.mean(sig ** 2)
    return float(energy / std)


# ==========================================================
# Heart Rate
# ==========================================================

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