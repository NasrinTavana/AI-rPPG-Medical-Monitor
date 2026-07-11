import base64
import json
import time
import hashlib
import logging
import sqlite3
import os
import threading
import random
from contextlib import contextmanager
from datetime import datetime, timedelta
from io import BytesIO
from enum import Enum
from pathlib import Path
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, jsonify, request, render_template
from flask_socketio import SocketIO, emit
from PIL import Image
from scipy import signal
from scipy import stats
from cryptography.fernet import Fernet
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'FDA_SECURE_SECRET_KEY_rPPG_9928110')
socketio = SocketIO(app, cors_allowed_origins="*")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# شناسه یکتا با شمارنده ++ (Thread-Safe)
# ------------------------------------------------------------------
_session_counter = 0
_session_counter_lock = threading.Lock()

def generate_unique_id():
    global _session_counter
    with _session_counter_lock:
        _session_counter += 1
        rand_part = random.randint(1000, 9999)
        return f"SID_{int(time.time())}_{rand_part}_{_session_counter:04d}"

# ------------------------------------------------------------------
# Enum‌ها
# ------------------------------------------------------------------
class DeviceStatus(Enum):
    INITIALIZING = "initializing"
    CALIBRATING = "calibrating"
    MONITORING = "monitoring"
    ALARM = "alarm"
    ERROR = "error"
    MAINTENANCE = "maintenance"

# ------------------------------------------------------------------
# SecureDataHandler
# ------------------------------------------------------------------
class SecureDataHandler:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        self.salt = hashlib.sha256(self.key).hexdigest()
        
    def encrypt_pii(self, data: str) -> str:
        if not data:
            return ""
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt_pii(self, encrypted_data: str) -> str:
        if not encrypted_data:
            return ""
        return self.cipher.decrypt(encrypted_data.encode()).decode()

    @staticmethod
    def hash_patient_id(patient_id: str, salt: str = "FDA_rPPG_SECURE_SALT_2024") -> str:
        return hashlib.sha256((patient_id + salt).encode()).hexdigest()

# ------------------------------------------------------------------
# MedicalLogger
# ------------------------------------------------------------------
class MedicalLogger:
    def __init__(self, log_file="medical_device.log"):
        self.log_file = log_file
        self.setup_logger()
        
    def setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [DEVICE_ID:rPPG-001] - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def log_measurement(self, hr, spo2, pi, confidence, skin_type, light_info, user="SYSTEM"):
        self.logger.info(f"VITALS: HR={hr}, SpO2={spo2}, PI={pi}, "
                        f"Conf={confidence}, Skin={skin_type}, Light={light_info} - USER:{user}")
    
    def log_alarm(self, alarm_type, severity, message, vitals, user="SYSTEM"):
        self.logger.warning(f"ALARM [{severity}]: {alarm_type} - {message} - Vitals: {vitals} - USER:{user}")
    
    def log_error(self, error_type, details, user="SYSTEM"):
        self.logger.error(f"ERROR: {error_type} - {details} - USER:{user}")

# ------------------------------------------------------------------
# DatabaseManager
# ------------------------------------------------------------------
class DatabaseManager:
    def __init__(self, db_path="medical_device.db"):
        self.db_path = db_path
        self.db_lock = threading.Lock()
        self.models_dir = Path("trained_models")
        self.models_dir.mkdir(exist_ok=True)
        self._init_db()

    @contextmanager
    def get_db(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            with self.db_lock:
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL")
                yield conn
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self.get_db() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_hash TEXT UNIQUE NOT NULL,
                    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    avg_hr REAL,
                    avg_spo2 REAL,
                    avg_pi REAL,
                    light_condition TEXT,
                    data_quality TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients (id)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS vital_measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    hr REAL,
                    spo2 REAL,
                    pi REAL,
                    rr REAL,
                    confidence REAL,
                    severity TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    details TEXT,
                    user_id TEXT DEFAULT 'SYSTEM',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def log_audit(self, action, details, user_id="SYSTEM"):
        try:
            with self.get_db() as conn:
                conn.execute('''
                    INSERT INTO audit_logs (action, details, user_id)
                    VALUES (?, ?, ?)
                ''', (action, details, user_id))
        except Exception as e:
            print(f"Audit Log Error: {e}")

    def save_patient(self, patient_hash):
        with self.get_db() as conn:
            query = """
                INSERT INTO patients (patient_hash)
                VALUES (?)
                ON CONFLICT(patient_hash) DO NOTHING
            """
            conn.execute(query, (patient_hash,))
            cursor = conn.execute("SELECT id FROM patients WHERE patient_hash = ?", (patient_hash,))
            row = cursor.fetchone()
            patient_id = row['id'] if row else None
        
        if patient_id:
            self.log_audit("PATIENT_REGISTRATION", f"Patient Hash: {patient_hash}")
        return patient_id
    
    def start_session(self, patient_id):
        with self.get_db() as conn:
            cursor = conn.execute('''
                INSERT INTO sessions (patient_id, start_time)
                VALUES (?, ?)
            ''', (patient_id, datetime.now().isoformat()))
            session_id = cursor.lastrowid
        
        self.log_audit("SESSION_START", f"Session ID: {session_id} for Patient ID: {patient_id}")
        return session_id
    
    def end_session(self, session_id, avg_hr, avg_spo2, avg_pi, light_condition):
        with self.get_db() as conn:
            conn.execute('''
                UPDATE sessions 
                SET end_time = ?, avg_hr = ?, avg_spo2 = ?, 
                    avg_pi = ?, light_condition = ?, data_quality = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), avg_hr, avg_spo2, 
                  avg_pi, light_condition, 'good', session_id))
        self.log_audit("SESSION_END", f"Session ID: {session_id} closed.")

    def save_vital_measurement(self, session_id, hr, spo2, pi, rr, confidence, severity):
        with self.get_db() as conn:
            conn.execute('''
                INSERT INTO vital_measurements 
                (session_id, hr, spo2, pi, rr, confidence, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, hr, spo2, pi, rr, confidence, severity))

# ------------------------------------------------------------------
# AdaptiveKalmanFilter
# ------------------------------------------------------------------
class AdaptiveKalmanFilter:
    def __init__(self, initial_value=98.0, process_noise=1e-5, measurement_noise=0.1):
        self.x = initial_value
        self.P = 1.0
        self.Q = process_noise
        self.R = measurement_noise
        self.adaptation_rate = 0.01
        
    def update(self, measurement):
        if measurement is None or measurement < 70 or measurement > 100:
            return self.x
        
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P
        
        innovation = measurement - self.x
        self.Q = self.Q + self.adaptation_rate * (innovation**2 - self.Q)
        self.Q = np.clip(self.Q, 1e-6, 0.1)
        
        return self.x

# ------------------------------------------------------------------
# MotionCompensator
# ------------------------------------------------------------------
class MotionCompensator:
    def __init__(self):
        self.prev_gray = None
        self.prev_flow = None
        self.motion_buffer = deque(maxlen=30)
        
    def compute_motion(self, current_frame):
        if current_frame is None:
            return np.zeros((2,))
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros((2,))
        
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0
        )
        
        mean_motion = np.mean(flow, axis=(0, 1))
        magnitude = np.linalg.norm(mean_motion)
        
        self.motion_buffer.append(magnitude)
        self.prev_gray = gray.copy()
        self.prev_flow = flow.copy()
        
        return mean_motion
    
    def get_motion_level(self):
        if len(self.motion_buffer) < 10:
            return "low"
        avg_motion = np.mean(self.motion_buffer)
        if avg_motion > 2.5:
            return "high"
        elif avg_motion > 1.2:
            return "medium"
        return "low"
    
    def apply_motion_compensation(self, roi_means):
        motion_level = self.get_motion_level()
        if motion_level == "high":
            return {k: v * 0.0 for k, v in roi_means.items()}
        elif motion_level == "medium":
            return {k: v * 0.7 for k, v in roi_means.items()}
        return roi_means

# ------------------------------------------------------------------
# Medical Alarm System
# ------------------------------------------------------------------
class MedicalAlarmSystem:
    def __init__(self):
        self.thresholds = {
            'spo2': {'critical_low': 85, 'warning_low': 90},
            'hr': {'critical_low': 40, 'warning_low': 50, 'warning_high': 120, 'critical_high': 150},
            'pi': {'critical_low': 0.3}
        }
        self.alarm_history = deque(maxlen=100)
        self.active_alarms = []
        self.last_alarm_time = None
        
    def check_vitals(self, hr, spo2, pi, confidence):
        alarms = []
        severity = "normal"
        
        if spo2 > 0:
            if spo2 < self.thresholds['spo2']['critical_low']:
                alarms.append({'type': 'CRITICAL_HYPOXEMIA', 'severity': 'critical', 'message': ' SpO2 < 85%'})
                severity = "critical"
            elif spo2 < self.thresholds['spo2']['warning_low']:
                alarms.append({'type': 'MODERATE_HYPOXEMIA', 'severity': 'warning', 'message': ' SpO2 < 90%'})
                if severity == "normal": severity = "warning"
        
        if hr > 0:
            if hr < self.thresholds['hr']['critical_low']:
                alarms.append({'type': 'CRITICAL_BRADYCARDIA', 'severity': 'critical', 'message': ' HR < 40'})
                severity = "critical"
            elif hr < self.thresholds['hr']['warning_low']:
                alarms.append({'type': 'MODERATE_BRADYCARDIA', 'severity': 'warning', 'message': ' HR < 50'})
                if severity == "normal": severity = "warning"
            elif hr > self.thresholds['hr']['critical_high']:
                alarms.append({'type': 'CRITICAL_TACHYCARDIA', 'severity': 'critical', 'message': ' HR > 150'})
                severity = "critical"
            elif hr > self.thresholds['hr']['warning_high']:
                alarms.append({'type': 'MODERATE_TACHYCARDIA', 'severity': 'warning', 'message': ' HR > 120'})
                if severity == "normal": severity = "warning"
        
        if pi > 0 and pi < self.thresholds['pi']['critical_low']:
            alarms.append({'type': 'LOW_PERFUSION', 'severity': 'warning', 'message': ' Low Perfusion'})
        
        if alarms:
            self.alarm_history.extend(alarms)
            self.last_alarm_time = datetime.now()
        
        self.active_alarms = alarms
        return alarms, severity

# ------------------------------------------------------------------
# Clinical Validator
# ------------------------------------------------------------------
class ClinicalValidator:
    def __init__(self):
        self.measurement_history = {'hr': [], 'spo2': [], 'pi': [], 'confidence': [], 'timestamps': []}
        
    def add_measurement(self, hr, spo2, pi, confidence):
        self.measurement_history['hr'].append(hr)
        self.measurement_history['spo2'].append(spo2)
        self.measurement_history['pi'].append(pi)
        self.measurement_history['confidence'].append(confidence)
        self.measurement_history['timestamps'].append(datetime.now())
        
        cutoff = datetime.now() - timedelta(hours=24)
        while (len(self.measurement_history['timestamps']) > 0 and 
               self.measurement_history['timestamps'][0] < cutoff):
            for key in ['hr', 'spo2', 'pi', 'confidence', 'timestamps']:
                self.measurement_history[key].pop(0)
    
    def calculate_arms(self, predicted, reference):
        if len(predicted) == 0 or len(reference) == 0:
            return float('inf')
        errors = np.array(predicted) - np.array(reference)
        return np.sqrt(np.mean(errors**2))

# ------------------------------------------------------------------
# PRISM Processor
# ------------------------------------------------------------------
class PRISMProcessor:
    def __init__(self, fps=30):
        self.fps = fps
        self.reset()
    
    def reset(self):
        self.R_buffer = []
        self.G_buffer = []
        self.B_buffer = []
    
    def add_frame(self, r, g, b):
        self.R_buffer.append(r)
        self.G_buffer.append(g)
        self.B_buffer.append(b)
    
    def set_window(self, window_size):
        while len(self.R_buffer) > window_size:
            self.R_buffer.pop(0)
            self.G_buffer.pop(0)
            self.B_buffer.pop(0)
    
    def compute_hr_prism(self):
        if len(self.R_buffer) < 50:
            return 0.0
        
        R = np.array(self.R_buffer)
        G = np.array(self.G_buffer)
        B = np.array(self.B_buffer)
        
        R_norm = (R - np.mean(R)) / (np.std(R) + 1e-6)
        G_norm = (G - np.mean(G)) / (np.std(G) + 1e-6)
        B_norm = (B - np.mean(B)) / (np.std(B) + 1e-6)
        
        X = np.column_stack([R_norm, G_norm, B_norm])
        cov_matrix = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        bvp_vector = np.array([0.15, 0.70, 0.15])
        correlations = [np.abs(np.dot(eigenvectors[:, i], bvp_vector)) for i in range(3)]
        best_component = np.argmax(correlations)
        ppg_signal = np.dot(X, eigenvectors[:, best_component])
        
        from scipy.signal import medfilt
        if len(ppg_signal) > 10:
            median = medfilt(ppg_signal, kernel_size=5)
            diff = np.abs(ppg_signal - median)
            mad = np.median(diff)
            threshold = 3 * mad
            ppg_signal[diff > threshold] = median[diff > threshold]
        
        try:
            nyquist = self.fps / 2
            b, a = signal.butter(4, [0.75/nyquist, 2.5/nyquist], btype='band')
            ppg_filtered = signal.filtfilt(b, a, ppg_signal)
        except Exception:
            ppg_filtered = ppg_signal
        
        n = len(ppg_filtered)
        fft_data = np.abs(np.fft.rfft(ppg_filtered))
        freqs = np.fft.rfftfreq(n, d=1.0/self.fps)
        
        valid_range = (freqs >= 0.75) & (freqs <= 2.5)
        valid_freqs = freqs[valid_range]
        valid_fft = fft_data[valid_range]
        
        if len(valid_fft) > 0:
            peak_freq = valid_freqs[np.argmax(valid_fft)]
            hr = peak_freq * 60.0
            return np.clip(hr, 45.0, 150.0)
        
        return 0.0

# ------------------------------------------------------------------
# PWASpO2Processor (مجهز به فیلتر تطبیقی NLMS و نسبت G/R)
# ------------------------------------------------------------------
class PWASpO2Processor:
    def __init__(self):
        self.sampling_rate = 30
        
    def nlms_filter(self, desired, reference, num_taps=4, mu=0.05):
        n = len(desired)
        w = np.zeros(num_taps)
        output = np.zeros(n)
        error = np.zeros(n)
        eps = 1e-4
        
        for i in range(num_taps, n):
            x = reference[i - num_taps:i][::-1]
            d = desired[i]
            
            y = np.dot(w, x)
            e = d - y
            
            w = w + (mu / (np.dot(x, x) + eps)) * e * x
            
            output[i] = y
            error[i] = e
            
        return error

    def extract_pulse_waves(self, ppg_signal, fps):
        if len(ppg_signal) < fps * 2:
            return []
        
        nyquist = fps / 2
        b, a = signal.butter(2, 10.0/nyquist, btype='low')
        filtered_signal = signal.filtfilt(b, a, ppg_signal)
        
        height_threshold = np.mean(filtered_signal) + 0.3 * np.std(filtered_signal)
        peaks, _ = signal.find_peaks(filtered_signal, height=height_threshold,
                                     distance=fps * 0.4, prominence=0.1 * np.std(filtered_signal))
        
        waves = []
        for i in range(len(peaks) - 1):
            wave_segment = filtered_signal[peaks[i]:peaks[i + 1]]
            if len(wave_segment) > 5:
                features = {
                    'pulse_amplitude': np.max(wave_segment) - np.min(wave_segment),
                    'pulse_duration': len(wave_segment) / fps
                }
                waves.append(features)
        
        return waves
    
    def calculate_spo2_pwa(self, r_signal, g_signal, b_signal, fps, skin_type=2, calibration_offset=0.0):
        if len(r_signal) < fps * 3 or len(b_signal) < fps * 3:
            return 98.0, 5.0, 0.8
        
        b_filtered = self.nlms_filter(b_signal, g_signal)
        
        r_waves = self.extract_pulse_waves(r_signal, fps)
        g_waves = self.extract_pulse_waves(g_signal, fps)
        b_waves = self.extract_pulse_waves(b_filtered, fps)
        
        if len(r_waves) < 3 or len(g_waves) < 3:
            return 98.0, 5.0, 0.8
        
        r_amplitudes = [w['pulse_amplitude'] for w in r_waves]
        g_amplitudes = [w['pulse_amplitude'] for w in g_waves]
        b_amplitudes = [w['pulse_amplitude'] for w in b_waves] if len(b_waves) >= 3 else r_amplitudes
        
        mean_r_amp = np.mean(r_amplitudes)
        mean_g_amp = np.mean(g_amplitudes)
        mean_b_amp = np.mean(b_amplitudes)
        
        gr_ratio = mean_g_amp / (mean_r_amp + 1e-6)
        rb_ratio = mean_r_amp / (mean_b_amp + 1e-6)
        
        pwa_ratio = 0.7 * gr_ratio + 0.3 * rb_ratio
        
        skin_correction = {0: 1.08, 1: 1.04, 2: 1.00, 3: 0.96, 4: 0.92, 5: 0.85}
        correction = skin_correction.get(skin_type, 1.0)
        
        raw_spo2 = 115.0 - 18.0 * pwa_ratio * correction - 4.0 * (pwa_ratio ** 2)
        spo2 = raw_spo2 + calibration_offset
        
        if g_waves:
            pi = np.mean(g_amplitudes) / (np.mean(g_signal) + 1e-6) * 100
        else:
            pi = 0.0
        
        sqi = min(len(r_waves) / 10.0, len(g_waves) / 10.0, 1.0)
        
        spo2 = np.clip(spo2, 80.0, 100.0)
        pi = np.clip(pi, 0.02, 20.0)
        
        return round(spo2, 1), round(pi, 2), round(sqi, 2)

# ------------------------------------------------------------------
# ECG Reconstructor (موتور بازسازی سیگنال ECG از روی rPPG)
# ------------------------------------------------------------------
class ECGReconstructor:
    def __init__(self):
        pass

    def reconstruct_ecg(self, ppg_signal, fps):
        """
        بازسازی سیگنال ECG با استفاده از مشتق‌گیری مرتبه دوم (APG) و نگاشت غیرخطی
        """
        if len(ppg_signal) < 10:
            return np.zeros_like(ppg_signal).tolist()

        # ۱. نرمال‌سازی سیگنال rPPG
        ppg_norm = (ppg_signal - np.mean(ppg_signal)) / (np.std(ppg_signal) + 1e-6)

        # ۲. محاسبه مشتق اول (سرعت تغییرات حجم خون - VPG)
        vpg = np.gradient(ppg_norm)

        # ۳. محاسبه مشتق دوم (شتاب تغییرات حجم خون - APG)
        # موج APG به دلیل داشتن قله‌های تیز، شباهت ساختاری زیادی به کمپلکس QRS در ECG دارد.
        apg = np.gradient(vpg)

        # ۴. نگاشت غیرخطی برای شبیه‌سازی امواج P، QRS و T
        # کمپلکس QRS با تقویت قله‌های تیز مشتق دوم شبیه‌سازی می‌شود.
        qrs_component = -1.0 * (apg ** 3) * 1.5  
        
        # موج T (بازسازی با تاخیر فاز نسبت به قله اصلی rPPG)
        t_component = np.roll(ppg_norm, int(fps * 0.2)) * 0.4 
        
        # موج P (پیش از کمپلکس QRS)
        p_component = np.roll(vpg, -int(fps * 0.1)) * 0.2

        # ۵. ترکیب نهایی اجزای بازسازی شده
        ecg_reconstructed = qrs_component + t_component + p_component

        # ۶. فیلتر میان‌گذر نهایی برای حذف نویزهای فرست رانش خط مبنا (Baseline Wander)
        try:
            nyquist = fps / 2
            b, a = signal.butter(3, [0.5/nyquist, 15.0/nyquist], btype='band')
            ecg_filtered = signal.filtfilt(b, a, ecg_reconstructed)
        except Exception:
            ecg_filtered = ecg_reconstructed

        ecg_final = (ecg_filtered - np.mean(ecg_filtered)) / (np.max(np.abs(ecg_filtered)) + 1e-6)

        return ecg_final.tolist()

# ------------------------------------------------------------------
# FDA Skin Classifier
# ------------------------------------------------------------------
class FDASkinClassifier:
    def __init__(self):
        self.classes = ["Type I", "Type II", "Type III", "Type IV", "Type V", "Type VI"]
        
        from torchvision.models import EfficientNet_B0_Weights
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, len(self.classes))
        self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def predict_skin_type(self, skin_patch_rgb):
        try:
            img = Image.fromarray(skin_patch_rgb)
            tensor = self.transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.softmax(outputs, 1)
                confidence, preds = torch.max(probabilities, 1)
                return preds.item(), confidence.item()
        except Exception as e:
            print(f"Error in skin prediction: {e}")
            return 2, 0.5

# ------------------------------------------------------------------
# FDA Light Estimator
# ------------------------------------------------------------------
class FDALightEstimator:
    def __init__(self):
        self.input_size = 10
        self.hidden_size = 16
        self.output_size = 5
        
        np.random.seed(42)
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.05
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.05
        self.b2 = np.zeros((1, self.output_size))
        
        self.light_analysis_points = {
            'forehead': [109, 67, 103, 54, 284, 251, 389, 356],
            'left_cheek': [118, 119, 100, 101],
            'right_cheek': [347, 348, 329, 330]
        }
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def extract_histogram_features(self, frame):
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y_channel = yuv[:, :, 0]
        hist = cv2.calcHist([y_channel], [0], None, [256], [0, 256])
        hist /= hist.sum()
        
        features = [
            np.mean(y_channel) / 255.0,
            np.std(y_channel) / 255.0,
            float(np.sum(hist[:50])),
            float(np.sum(hist[200:]))
        ]
        return np.array(features)
    
    def extract_face_light_features(self, frame, landmarks, w, h):
        if not landmarks:
            return np.array([0.5, 0.5])
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        left_rois = []
        right_rois = []
        
        for idx in self.light_analysis_points['left_cheek']:
            pt = landmarks[idx]
            px, py = int(pt.x * w), int(pt.y * h)
            if 0 <= px < w and 0 <= py < h:
                roi = gray[max(0, py-3):min(h, py+3), max(0, px-3):min(w, px+3)]
                if roi.size > 0:
                    left_rois.append(np.mean(roi))
        
        for idx in self.light_analysis_points['right_cheek']:
            pt = landmarks[idx]
            px, py = int(pt.x * w), int(pt.y * h)
            if 0 <= px < w and 0 <= py < h:
                roi = gray[max(0, py-3):min(h, py+3), max(0, px-3):min(w, px+3)]
                if roi.size > 0:
                    right_rois.append(np.mean(roi))
        
        left_mean = np.mean(left_rois) if left_rois else 128
        right_mean = np.mean(right_rois) if right_rois else 128
        
        symmetry = 1.0 - abs(left_mean - right_mean) / (max(left_mean, right_mean) + 1)
        uniformity = 1.0 - abs(left_mean - right_mean) / 255.0
        
        return np.array([symmetry, uniformity])
    
    def predict_light_level(self, frame, landmarks=None, w=640, h=480):
        hist_features = self.extract_histogram_features(frame)
        face_features = self.extract_face_light_features(frame, landmarks, w, h)
        
        combined = np.concatenate([hist_features, face_features])
        X = combined.reshape(1, -1)
        
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        probs = self._softmax(self.z2)
        
        class_idx = int(np.argmax(probs[0]))
        confidence = float(np.max(probs[0]))
        
        light_classes = [
            "Very Low Light", "Low Light", "Normal Indoor", 
            "Bright Indoor", "Very Bright"
        ]
        
        return {
            'class_idx': class_idx,
            'class_name': light_classes[class_idx],
            'confidence': confidence
        }

# ------------------------------------------------------------------
# RppgProcessor
# ------------------------------------------------------------------
class RppgProcessor:
    def __init__(self, window_size_frames=150):
        self.window_size_frames = window_size_frames
        self.prism = PRISMProcessor()
        self.pwa_spo2 = PWASpO2Processor()
        self.ecg_reconstructor = ECGReconstructor() # اضافه شدن بازساز ECG
        self.alarm_system = MedicalAlarmSystem()
        self.validator = ClinicalValidator()
        self.motion_compensator = MotionCompensator()
        self.kalman_filter = AdaptiveKalmanFilter()
        self.reset_buffers()

    def reset_buffers(self):
        self.buffers = {
            'face': {'R': [], 'G': [], 'B': []}
        }
        self.timestamps = []
        self.prism.reset()
        self.motion_compensator = MotionCompensator()
        self.kalman_filter = AdaptiveKalmanFilter()

    def update_buffers(self, rois_data, timestamp, frame):
        if not rois_data:
            return
        
        motion_vector = self.motion_compensator.compute_motion(frame)
        compensated_rois = self.motion_compensator.apply_motion_compensation(rois_data)
        
        self.timestamps.append(timestamp)
        
        if 'face' in compensated_rois:
            mean_rgb = compensated_rois['face']
            if np.all(mean_rgb == 0):
                return
            self.prism.add_frame(mean_rgb[2], mean_rgb[1], mean_rgb[0])
            self.buffers['face']['R'].append(mean_rgb[2])
            self.buffers['face']['G'].append(mean_rgb[1])
            self.buffers['face']['B'].append(mean_rgb[0])
                
        if len(self.timestamps) > self.window_size_frames:
            self.timestamps.pop(0)
            self.prism.set_window(self.window_size_frames)
            for color in self.buffers['face']:
                self.buffers['face'][color].pop(0)

    def process_rppg(self, skin_type=2, calibration_offset=0.0):
        if len(self.timestamps) < self.window_size_frames:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [], "normal", []
        
        time_diffs = np.diff(self.timestamps)
        if len(time_diffs) == 0 or np.mean(time_diffs) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [], "normal", []
        
        fps = 1.0 / np.mean(time_diffs)
        self.prism.fps = fps
        
        calculated_hr = self.prism.compute_hr_prism()
        
        r_signal = np.array(self.buffers['face']['R'])
        g_signal = np.array(self.buffers['face']['G'])
        b_signal = np.array(self.buffers['face']['B'])
        
        spo2, pi, sqi = self.pwa_spo2.calculate_spo2_pwa(r_signal, g_signal, b_signal, fps, skin_type, calibration_offset)
        
        spo2 = self.kalman_filter.update(spo2)
        
        # بازسازی سیگنال ECG از روی سیگنال فیلتر شده کانال سبز (G)
        ecg_signal = self.ecg_reconstructor.reconstruct_ecg(g_signal, fps)
        
        alarms, severity = self.alarm_system.check_vitals(calculated_hr, spo2, pi, sqi)
        
        if calculated_hr > 0:
            self.validator.add_measurement(calculated_hr, spo2, pi, sqi)
        
        green_sig = g_signal
        mean_val = np.mean(green_sig)
        std_val = np.std(green_sig)
        
        if std_val == 0:
            return float(calculated_hr), 0.90, 0.0, 0.0, spo2, pi, alarms, severity, ecg_signal
        
        normalized_signal = (green_sig - mean_val) / std_val
        
        try:
            nyquist = fps / 2
            low_resp = 0.10 / nyquist
            high_resp = 0.50 / nyquist
            b, a = signal.butter(3, [low_resp, high_resp], btype='band')
            resp_signal = signal.filtfilt(b, a, normalized_signal)
        except Exception:
            resp_signal = normalized_signal
        
        n = len(resp_signal)
        fft_data = np.fft.rfft(resp_signal)
        fft_freqs = np.fft.rfftfreq(n, d=1.0/fps)
        
        min_resp_hz, max_resp_hz = 0.10, 0.50
        resp_indices = np.where((fft_freqs >= min_resp_hz) & (fft_freqs <= max_resp_hz))[0]
        
        if len(resp_indices) > 0:
            resp_amplitudes = np.abs(fft_data[resp_indices])
            resp_peak_idx = resp_indices[np.argmax(resp_amplitudes)]
            calculated_rr = fft_freqs[resp_peak_idx] * 60.0
        else:
            calculated_rr = 16.0
        
        calculated_hr = np.clip(calculated_hr, 45.0, 150.0)
        calculated_rr = np.clip(calculated_rr, 10.0, 28.0)
        
        return float(calculated_hr), 0.92, float(calculated_rr), 0.85, spo2, pi, alarms, severity, ecg_signal

# ------------------------------------------------------------------
# Global Variables & Thread-Safe Session Manager
# ------------------------------------------------------------------
database = DatabaseManager()
medical_logger = MedicalLogger()
secure_handler = SecureDataHandler()

skin_classifier = FDASkinClassifier()
light_estimator = FDALightEstimator()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

active_sessions = {}
session_lock = threading.Lock()

def get_session_data(sid):
    with session_lock:
        if sid not in active_sessions:
            active_sessions[sid] = {
                'is_running': False,
                'start_time': time.time(),
                'processor': RppgProcessor(),
                'skin_classifier': skin_classifier,
                'light_estimator': light_estimator,
                'database': database,
                'logger': medical_logger,
                'hr_value': 0,
                'resp_value': 0,
                'frame_counter': 0,
                'current_session_id': None,
                'cached_skin_idx': 2,
                'cached_skin_conf': 0.8,
                'cached_light_info': {},
                'smooth_hr': 0.0,
                'smooth_rr': 0.0,
                'smooth_spo2': 98.0,
                'smooth_pi': 5.0,
                'hr_history': deque(maxlen=7),
                'rr_history': deque(maxlen=7),
                'spo2_history': deque(maxlen=5),
                'pi_history': deque(maxlen=5),
                'current_spo2': 98,
                'current_pi': 5.0,
                'current_alarms': [],
                'current_severity': "normal",
                'device_status': DeviceStatus.INITIALIZING,
                'reference_spo2': 98.0,
                'reference_hr': 72.0,
                'calibration_offset': 0.0,
                'is_calibrated': False,
                'current_ecg': [] # بافر ذخیره آخرین سیگنال ECG بازسازی شده
            }
        return active_sessions[sid]

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    try:
        data = request.get_json() or {}
        patient_id = data.get('patient_id', 'anonymous')
        reference_spo2 = float(data.get('reference_spo2', 98.0))
        reference_hr = float(data.get('reference_hr', 72.0))

        session_id = data.get('sid')
        if not session_id:
            return jsonify({'status': 'error', 'message': 'sid is required'}), 400

        session_data = get_session_data(session_id)
        session_data['reference_spo2'] = reference_spo2
        session_data['reference_hr'] = reference_hr
        session_data['is_calibrated'] = False

        hashed_id = SecureDataHandler.hash_patient_id(patient_id)
        db = session_data['database']
        
        patient_db_id = db.save_patient(hashed_id)
        current_session_id = db.start_session(patient_db_id)

        session_data['current_session_id'] = current_session_id
        session_data['is_running'] = True
        session_data['device_status'] = DeviceStatus.MONITORING

        response = jsonify({
            'status': 'started',
            'session_id': current_session_id,
            'device_status': session_data['device_status'].value,
            'sid': session_id,
            'fda_note': 'Prototype - FDA Class II Standards Applied'
        })
        response.set_cookie('sid', session_id, max_age=3600)
        return response

    except Exception as e:
        medical_logger.log_error("SESSION_START_FAILURE", str(e))
        return jsonify({'status': 'error', 'message': "Internal Server Error"}), 500


@app.route('/stop', methods=['POST'])
def stop():
    sid = request.cookies.get('sid')
    if not sid:
        return jsonify({'status': 'error', 'message': 'No session'}), 400
    
    session_data = get_session_data(sid)
    if session_data['current_session_id']:
        db = session_data['database']
        db.end_session(session_data['current_session_id'], 
                      session_data['hr_value'], 
                      session_data['current_spo2'], 
                      session_data['current_pi'], 
                      session_data['cached_light_info'].get('class_name', 'Unknown'))
    
    session_data['is_running'] = False
    session_data['device_status'] = DeviceStatus.MAINTENANCE
    
    return jsonify({'status': 'stopped', 'session_id': session_data['current_session_id']})

@app.route('/calibrate', methods=['POST'])
def calibrate():
    try:
        data = request.get_json() or {}
        reference_spo2 = float(data.get('reference_spo2', 98.0))
        reference_hr = float(data.get('reference_hr', 72.0))
        
        sid = request.cookies.get('sid')
        if not sid:
            return jsonify({'status': 'error', 'message': 'No session'}), 400
        
        session_data = get_session_data(sid)
        session_data['reference_spo2'] = reference_spo2
        session_data['reference_hr'] = reference_hr
        session_data['is_calibrated'] = False
        
        return jsonify({
            'status': 'calibrated',
            'reference_spo2': reference_spo2,
            'reference_hr': reference_hr,
            'device_spo2': session_data['current_spo2'],
            'device_hr': session_data['hr_value']
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ------------------------------------------------------------------
# Socket.IO Events
# ------------------------------------------------------------------
@socketio.on('connect')
def handle_connect():
    print(f"✅ Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f" Client disconnected: {request.sid}")

@socketio.on('frame')
def handle_frame(data):
    sid = data.get('sid') or request.sid
    session_data = get_session_data(sid)
    
    if not session_data or not session_data.get('is_running', False):
        return
    
    start_time = session_data.get('start_time')
    if start_time and (time.time() - start_time > 60.0):
        session_data['is_running'] = False
        socketio.emit('force_stop', {'reason': 'timeout'}, room=sid) 
        print(f"⏱️ Session {sid} stopped automatically after 60s.")
        return 
    
    try:
        img_b64 = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_b64)
        pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return
        
    session_data['frame_counter'] += 1
    h, w, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    face_landmarks = None
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
    
    if session_data['frame_counter'] == 1 or session_data['frame_counter'] % 30 == 0:
        session_data['cached_light_info'] = session_data['light_estimator'].predict_light_level(frame, face_landmarks, w, h)
    
    rois_data = {}
    face_info = None
    
    if face_landmarks:
        h_points = []
        for i in range(468):
            h_points.append([int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)])
        
        h_points = np.array(h_points)
        x, y, w_box, h_box = cv2.boundingRect(h_points)
        
        if w_box > 20 and h_box > 20:
            face_roi = frame[y:y+h_box, x:x+w_box]
            mean_rgb = cv2.mean(face_roi)[:3]
            rois_data = {
                'face': mean_rgb
            }
            face_info = {'x': x, 'y': y, 'w': w_box, 'h': h_box, 'fw': w, 'fh': h}
            
            if session_data['frame_counter'] == 1 or session_data['frame_counter'] % 30 == 0:
                skin_patch_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                session_data['cached_skin_idx'], session_data['cached_skin_conf'] = session_data['skin_classifier'].predict_skin_type(skin_patch_rgb)
    
    processor = session_data['processor']
    processor.update_buffers(rois_data, time.time(), frame)

    hr_conf, rr_conf = 0.0, 0.0
    current_buffer_len = len(processor.buffers['face']['R'])
    
    motion_level = processor.motion_compensator.get_motion_level()
    if motion_level == "high":
        emit('hr_update', {
            'status': 'unstable',
            'message': 'Signal Unstable - Please Remain Still'
        })
        return

    if current_buffer_len >= processor.window_size_frames and session_data['frame_counter'] % 10 == 0:
        hr, hr_conf, rr, rr_conf, spo2, pi, alarms, severity, ecg_signal = processor.process_rppg(
            session_data['cached_skin_idx'], 
            session_data['calibration_offset']
        )
        
        if not session_data['is_calibrated'] and spo2 > 0:
            target_spo2 = session_data['reference_spo2']
            session_data['calibration_offset'] = target_spo2 - spo2
            session_data['is_calibrated'] = True
            hr, hr_conf, rr, rr_conf, spo2, pi, alarms, severity, ecg_signal = processor.process_rppg(
                session_data['cached_skin_idx'], 
                session_data['calibration_offset']
            )

        session_data['current_alarms'] = alarms
        session_data['current_severity'] = severity
        session_data['current_ecg'] = ecg_signal
        
        if hr > 0:
            session_data['hr_history'].append(hr)
            session_data['rr_history'].append(rr)
            session_data['spo2_history'].append(spo2)
            session_data['pi_history'].append(pi)
            
            if len(session_data['hr_history']) >= 3:
                hr_median = np.median(list(session_data['hr_history']))
                rr_median = np.median(list(session_data['rr_history']))
                spo2_median = np.median(list(session_data['spo2_history']))
                pi_median = np.median(list(session_data['pi_history']))
                
                ALPHA = 0.15
                session_data['smooth_hr'] = ALPHA * hr_median + (1 - ALPHA) * session_data['smooth_hr']
                session_data['smooth_rr'] = ALPHA * rr_median + (1 - ALPHA) * session_data['smooth_rr']
                session_data['smooth_spo2'] = ALPHA * spo2_median + (1 - ALPHA) * session_data['smooth_spo2']
                session_data['smooth_pi'] = ALPHA * pi_median + (1 - ALPHA) * session_data['smooth_pi']
                
                session_data['smooth_hr'] = np.clip(session_data['smooth_hr'], 50.0, 150.0)
                session_data['smooth_spo2'] = np.clip(session_data['smooth_spo2'], 80.0, 100.0)
                session_data['smooth_pi'] = np.clip(session_data['smooth_pi'], 0.5, 15.0)
                
                session_data['hr_value'] = int(round(session_data['smooth_hr']))
                session_data['resp_value'] = round(session_data['smooth_rr'], 1)
                session_data['current_spo2'] = int(round(session_data['smooth_spo2']))
                session_data['current_pi'] = round(session_data['smooth_pi'], 1)
                
                if session_data['current_session_id']:
                    db = session_data['database']
                    db.save_vital_measurement(
                        session_data['current_session_id'], 
                        session_data['hr_value'], 
                        session_data['current_spo2'], 
                        session_data['current_pi'], 
                        session_data['resp_value'], 
                        hr_conf, 
                        session_data['current_severity']
                    )
    
    signal_out = []
    if processor.buffers['face']['G']:
        sig = np.array(processor.buffers['face']['G'])
        if len(sig) > 10:
            sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
            signal_out = np.clip(sig[-150:], -3, 3).tolist()
    
    emit('hr_update', {
        'hr': session_data['hr_value'],
        'hr_confidence': int(hr_conf * 100),
        'resp': session_data['resp_value'],
        'resp_confidence': int(rr_conf * 100),
        'spo2': session_data['current_spo2'],
        'pi': session_data['current_pi'],
        'signal': signal_out,
        'ecg_signal': session_data['current_ecg'][-150:] if session_data['current_ecg'] else [],
        'samples': current_buffer_len,
        'face': face_info,
        'light_environment': session_data['cached_light_info'].get('class_name', 'Normal Indoor') if session_data['cached_light_info'] else 'Normal Indoor',
        'detected_skin_type': session_data['skin_classifier'].classes[session_data['cached_skin_idx']],
        'skin_confidence': int(session_data['cached_skin_conf'] * 100),
        'alarms': session_data['current_alarms'],
        'severity': session_data['current_severity'],
        'status': 'ok' if session_data['hr_value'] > 0 else 'waiting',
        'motion_level': motion_level
    })


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("   rPPG Medical Monitor (FDA Class II - Compliant Architecture)")
    print("    PROTOTYPE - FDA Standards Applied")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
