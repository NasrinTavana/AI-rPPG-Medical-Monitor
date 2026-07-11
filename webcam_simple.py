import os
import time
import json
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# ==========================================
# ۱. کلاس پردازش سیگنال و استخراج rPPG
# ==========================================
class RPPGProcessor:
    def __init__(self, fps=30, window_size_sec=15):
        self.fps = fps
        self.window_size_frames = int(window_size_sec * fps)
        
        # بافرهای ذخیره سیگنال‌های خام میانگین کانال‌های رنگی برای ۳ ناحیه پیشانی، گونه چپ و گونه راست
        self.buffers = {
            'forehead': {'R': [], 'G': [], 'B': [], 'timestamps': []},
            'left_cheek': {'R': [], 'G': [], 'B': [], 'timestamps': []},
            'right_cheek': {'R': [], 'G': [], 'B': [], 'timestamps': []}
        }
        
        # بافر میکروحرکات چهره برای تخمین تنفس (موقعیت Y پیشانی)
        self.y_motion_buffer = []

    def update_buffers(self, rois_data, timestamp):
        """
        rois_data: دیکشنری حاوی مقادیر میانگین RGB برای هر ناحیه و موقعیت Y پیشانی
        """
        for roi_name, rgb_vals in rois_data.items():
            if roi_name in self.buffers and rgb_vals is not None:
                self.buffers[roi_name]['R'].append(rgb_vals[0])
                self.buffers[roi_name]['G'].append(rgb_vals[1])
                self.buffers[roi_name]['B'].append(rgb_vals[2])
                self.buffers[roi_name]['timestamps'].append(timestamp)
                
                # نگه‌داشتن اندازه بافر در محدوده پنجره زمانی
                if len(self.buffers[roi_name]['R']) > self.window_size_frames:
                    self.buffers[roi_name]['R'].pop(0)
                    self.buffers[roi_name]['G'].pop(0)
                    self.buffers[roi_name]['B'].pop(0)
                    self.buffers[roi_name]['timestamps'].pop(0)
        
        if 'forehead_y' in rois_data and rois_data['forehead_y'] is not None:
            self.y_motion_buffer.append(rois_data['forehead_y'])
            if len(self.y_motion_buffer) > self.window_size_frames:
                self.y_motion_buffer.pop(0)

    def _butter_bandpass(self, lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def _apply_filter(self, data, lowcut, highcut):
        if len(data) < 15:
            return data
        b, a = self._butter_bandpass(lowcut, highcut, self.fps, order=2)
        return filtfilt(b, a, data)

    def _chrom_method(self, R, G, B):
        """پیاده‌سازی الگوریتم Chrominance-based (CHROM)"""
        R_f = self._apply_filter(R, 0.7, 4.0)
        G_f = self._apply_filter(G, 0.7, 4.0)
        B_f = self._apply_filter(B, 0.7, 4.0)
        
        # نرمال‌سازی با میانگین متحرک
        mean_R, mean_G, mean_B = np.mean(R), np.mean(G), np.mean(B)
        if mean_R == 0 or mean_G == 0 or mean_B == 0:
            return np.zeros_like(R)
            
        X = 3 * R_f / mean_R - 2 * G_f / mean_G
        Y = 1.5 * R_f / mean_R + G_f / mean_G - 1.5 * B_f / mean_B
        
        # ترکیب کرومینانس‌ها
        alpha = np.std(X) / (np.std(Y) + 1e-6)
        bvp = X - alpha * Y
        return bvp

    def _pos_method(self, R, G, B):
        """پیاده‌سازی الگوریتم Plane-Orthogonal-to-Skin (POS)"""
        H = np.array([R, G, B])
        if H.shape[1] < 10:
            return np.zeros_like(R)
            
        mean_H = np.mean(H, axis=1, keepdims=True)
        Cn = H / (mean_H + 1e-6)
        
        S = np.array([
            [0, 1, -1],
            [-2, 1, 1]
        ])
        P = S @ Cn
        
        alpha = np.std(P[0, :]) / (np.std(P[1, :]) + 1e-6)
        bvp = P[0, :] - alpha * P[1, :]
        return bvp

    def _calculate_hr_from_bvp(self, bvp):
        """محاسبه ضربان قلب با روش ولچ (Welch PSD) و تخمین معیار اعتماد طیفی (SQI)"""
        if len(bvp) < self.window_size_frames:
            return 0.0, 0.0
            
        # فیلترینگ نهایی در محدوده ضربان قلب انسان (45 تا 180 BPM)
        bvp_filtered = self._apply_filter(bvp, 0.75, 3.0) 
        
        fs = self.fps
        nperseg = min(len(bvp_filtered), 256)
        freqs, psd = welch(bvp_filtered, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        
        # محدود کردن به فرکانس‌های مجاز فیزیولوژیک
        valid_idx = (freqs >= 0.75) & (freqs <= 3.0)
        if not np.any(valid_idx):
            return 0.0, 0.0
            
        freqs_valid = freqs[valid_idx]
        psd_valid = psd[valid_idx]
        
        peak_freq = freqs_valid[np.argmax(psd_valid)]
        hr_bpm = peak_freq * 60.0
        
        # محاسبه Confidence (نسبت توان پیک به توان کل باند طیفی ضربان)
        peak_idx = np.argmax(psd_valid)
        band_power = np.sum(psd_valid)
        
        # تعریف یک پنجره کوچک دور پیک اصلی
        low_limit = max(0, peak_idx - 2)
        high_limit = min(len(psd_valid) - 1, peak_idx + 2)
        peak_power = np.sum(psd_valid[low_limit:high_limit+1])
        
        confidence = peak_power / (band_power + 1e-6)
        return hr_bpm, confidence

    def process_rppg(self):
        """پردازش اصلی و ادغام سیگنال‌های ROIهای مختلف"""
        hr_estimates = []
        confidences = []
        
        # پردازش جداگانه هر ناحیه و استفاده از الگوریتم بهینه
        for roi in ['forehead', 'left_cheek', 'right_cheek']:
            R = np.array(self.buffers[roi]['R'])
            G = np.array(self.buffers[roi]['G'])
            B = np.array(self.buffers[roi]['B'])
            
            if len(R) < self.window_size_frames:
                continue
                
            # اعمال دو روش CHROM و POS
            bvp_chrom = self._chrom_method(R, G, B)
            hr_c, conf_c = self._calculate_hr_from_bvp(bvp_chrom)
            
            bvp_pos = self._pos_method(R, G, B)
            hr_p, conf_p = self._calculate_hr_from_bvp(bvp_pos)
            
            # انتخاب الگوریتم بهتر برای این ROI بر اساس Confidence بالاتر
            if conf_c > conf_p:
                hr_estimates.append(hr_c)
                confidences.append(conf_c)
            else:
                hr_estimates.append(hr_p)
                confidences.append(conf_p)
                
        if not hr_estimates:
            return 0.0, 0.0, 0.0, 0.0
            
        # ترکیب تصمیم‌گیری بر اساس ضریب اطمینان (Weighted Mean)
        weights = np.array(confidences)
        total_w = np.sum(weights)
        if total_w > 0:
            final_hr = np.sum(np.array(hr_estimates) * weights) / total_w
            final_confidence = np.mean(weights)
        else:
            final_hr, final_confidence = 0.0, 0.0
            
        # محاسبه نرخ تنفس (RR) بر اساس سیگنال‌های میکروحرکت سر
        final_rr, rr_confidence = self._process_respiration()
        
        return final_hr, final_confidence, final_rr, rr_confidence

    def _process_respiration(self):
        """محاسبه نرخ تنفس از تغییرات موقعیت عمودی سر (Y axis)"""
        if len(self.y_motion_buffer) < self.window_size_frames:
            return 0.0, 0.0
            
        motion_data = np.array(self.y_motion_buffer)
        
        # فیلتر تنفس انسان (بین 10 تا 30 تنفس در دقیقه یا 0.15 تا 0.5 هرتز)
        motion_filtered = self._apply_filter(motion_data, 0.15, 0.5)
        
        nperseg = min(len(motion_filtered), 256)
        freqs, psd = welch(motion_filtered, fs=self.fps, nperseg=nperseg, noverlap=nperseg//2)
        
        valid_idx = (freqs >= 0.15) & (freqs <= 0.5)
        if not np.any(valid_idx):
            return 0.0, 0.0
            
        freqs_valid = freqs[valid_idx]
        psd_valid = psd[valid_idx]
        
        peak_freq = freqs_valid[np.argmax(psd_valid)]
        rr_bpm = peak_freq * 60.0
        
        band_power = np.sum(psd_valid)
        peak_power = np.max(psd_valid)
        rr_confidence = peak_power / (band_power + 1e-6)
        
        return rr_bpm, rr_confidence


# ==========================================
# ۲. کلاس ذخیره داده‌ها و اعتبارسنجی
# ==========================================
class DataCollector:
    def __init__(self, output_dir="rppg_outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.session_id = time.strftime("session_%Y%m%d_%H%M%S")
        self.log_file = os.path.join(output_dir, f"{self.session_id}_data.csv")
        self.metadata_file = os.path.join(output_dir, f"{self.session_id}_metadata.json")
        self.records = []
        
    def collect_demographics(self):
        """دریافت اطلاعات دموگرافیک از کاربر پیش از اجرای پایپ‌لاین"""
        print("\n=== مشخصات دموگرافیک و محیط کاربری ===")
        age = input("سن کاربر (مثال: 28): ")
        gender = input("جنسیت (Male/Female/Other): ")
        skin_type = input("نوع پوست فیتزپاتریک (1 تا 6): ")
        lux = input("شدت نور محیط به لوکس (حدودی - اختیاری): ")
        
        self.metadata = {
            "session_id": self.session_id,
            "timestamp": time.time(),
            "demographics": {
                "age": age,
                "gender": gender,
                "fitzpatrick_skin_type": skin_type
            },
            "environment": {
                "estimated_lux": lux
            }
        }
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=4)

    def log_record(self, timestamp, frame_idx, hr, hr_conf, rr, rr_conf, gt_hr, gt_rr):
        record = {
            "timestamp": timestamp,
            "frame_index": frame_idx,
            "rppg_hr": hr,
            "rppg_hr_confidence": hr_conf,
            "rppg_rr": rr,
            "rppg_rr_confidence": rr_conf,
            "ground_truth_hr": gt_hr,
            "ground_truth_rr": gt_rr
        }
        self.records.append(record)
        
    def save_csv(self):
        if not self.records:
            return
        df = pd.DataFrame(self.records)
        df.to_csv(self.log_file, index=False)
        print(f"\n[INFO] داده‌های خام جلسه در فایل روبرو ذخیره شد:\n{self.log_file}")

    def generate_statistics_and_plots(self):
        """انجام محاسبات آماری دقیق و ترسیم نمودار Bland-Altman"""
        if not self.records:
            print("داده‌ای برای ارزیابی ثبت نشده است.")
            return
            
        df = pd.DataFrame(self.records)
        # فیلتر کردن مقادیر غیرصفر جهت مقایسه آماری منطقی
        df_valid = df[(df['rppg_hr'] > 0) & (df['ground_truth_hr'] > 0)]
        
        if len(df_valid) < 5:
            print("[WARNING] برای محاسبه آمار تعداد داده‌های معتبر ثبت‌شده بسیار کم است.")
            return

        # محاسبات آماری
        diff = df_valid['rppg_hr'] - df_valid['ground_truth_hr']
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff**2))
        mean_bias = np.mean(diff)
        std_diff = np.std(diff)
        loa_upper = mean_bias + 1.96 * std_diff
        loa_lower = mean_bias - 1.96 * std_diff
        
        print("\n================ نتایج ارزیابی آماری ================")
        print(f"تعداد کل فریم‌های معتبر محاسباتی: {len(df_valid)}")
        print(f"میانگین خطای مطلق (MAE): {mae:.2f} BPM")
        print(f"ریشه میانگین مربعات خطا (RMSE): {rmse:.2f} BPM")
        print(f"خطای سیستماتیک (Bland-Altman Bias): {mean_bias:.2f} BPM")
        print(f"بازه توافق (LoA 95%): [{loa_lower:.2f} to {loa_upper:.2f}] BPM")
        print("====================================================")

        # رسم نمودارها
        plt.figure(figsize=(12, 5))

        # ۱. نمودار مقایسه زمانی
        plt.subplot(1, 2, 1)
        plt.plot(df_valid['timestamp'] - df_valid['timestamp'].iloc[0], df_valid['ground_truth_hr'], 'g-', label='Ground Truth')
        plt.plot(df_valid['timestamp'] - df_valid['timestamp'].iloc[0], df_valid['rppg_hr'], 'r--', label='rPPG Estimation')
        plt.title('HR Tracking Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('HR (BPM)')
        plt.legend()
        plt.grid(True)

        # ۲. نمودار Bland-Altman
        plt.subplot(1, 2, 2)
        mean_val = (df_valid['rppg_hr'] + df_valid['ground_truth_hr']) / 2.0
        plt.scatter(mean_val, diff, color='blue', alpha=0.6, edgecolors='k')
        plt.axhline(mean_bias, color='red', linestyle='-', label=f'Bias: {mean_bias:.2f}')
        plt.axhline(loa_upper, color='gray', linestyle='--', label=f'+1.96 SD: {loa_upper:.2f}')
        plt.axhline(loa_lower, color='gray', linestyle='--', label=f'-1.96 SD: {loa_lower:.2f}')
        plt.title('Bland-Altman Plot')
        plt.xlabel('Mean of rPPG and Ground Truth (BPM)')
        plt.ylabel('Difference (rPPG - GT) (BPM)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"{self.session_id}_plots.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[INFO] نمودارهای ارزیابی در فایل زیر ذخیره شدند:\n{plot_path}")


# ==========================================
# ۳. اجرای حلقه بلادرنگ (Main Routine)
# ==========================================
def main():
    # ساخت شبیه‌ساز Ground Truth یا لود داده‌های فیزیکی
    # در اینجا برای تست مستقل، یک نوسان‌ساز ساده متناسب با ضربان قلب می‌نویسیم
    def get_ground_truth(t):
        # شبیه‌سازی ضربان قلب حول ۷۲ و تنفس حول ۱۶ با نوسانات ملایم
        sim_hr = 72.0 + 3.0 * np.sin(t * 0.1)
        sim_rr = 16.0 + 1.0 * np.cos(t * 0.05)
        return sim_hr, sim_rr

    collector = DataCollector()
    collector.collect_demographics()
    
    # راه‌اندازی وب‌کم و مدل‌های تشخیص چهره مدی پایپ
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] قادر به دسترسی به وب‌کم نیستیم.")
        return
        
    fps = 30 # فرکانس پیش‌فرض
    processor = RPPGProcessor(fps=fps, window_size_sec=15)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    print("\n[START] پردازش زنده آغاز شد. جهت خروج و استخراج آمار کلید 'q' را فشار دهید.")
    frame_idx = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = time.time()
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        rois_data = {}
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # مختصات لندمارک‌های صورت برای ROIها
            # پیشانی
            forehead_idx = [109, 67, 103, 54, 284, 251, 389, 356]
            # گونه چپ
            l_cheek_idx = [118, 119, 100, 101, 50, 205, 207]
            # گونه راست
            r_cheek_idx = [347, 348, 329, 330, 280, 425, 427]
            
            def get_roi_mean_rgb(idx_list):
                points = []
                for idx in idx_list:
                    pt = landmarks[idx]
                    points.append([int(pt.x * w), int(pt.y * h)])
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(points)], 255)
                mean_val = cv2.mean(frame, mask=mask)[:3]
                return mean_val, points
            
            f_rgb, f_pts = get_roi_mean_rgb(forehead_idx)
            lc_rgb, lc_pts = get_roi_mean_rgb(l_cheek_idx)
            rc_rgb, rc_pts = get_roi_mean_rgb(r_cheek_idx)
            
            # ذخیره لندمارک Y پیشانی برای میکروحرکات تنفس
            forehead_y = landmarks[10].y * h
            
            rois_data = {
                'forehead': f_rgb,
                'left_cheek': lc_rgb,
                'right_cheek': rc_rgb,
                'forehead_y': forehead_y
            }
            
            # کشیدن پلی‌گون‌ها روی تصویر زنده جهت اطمینان از ROI
            cv2.polylines(frame, [np.array(f_pts)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [np.array(lc_pts)], True, (255, 0, 0), 1)
            cv2.polylines(frame, [np.array(rc_pts)], True, (0, 0, 255), 1)
        
        # بروزرسانی بافرهای سیگنال
        processor.update_buffers(rois_data, current_time)
        
        # اجرای محاسبات در صورت پر شدن پنجره
        hr, hr_conf, rr, rr_conf = 0.0, 0.0, 0.0, 0.0
        if frame_idx >= processor.window_size_frames:
            hr, hr_conf, rr, rr_conf = processor.process_rppg()
            
        # دریافت داده‌های همزمان Ground Truth مرجع
        gt_hr, gt_rr = get_ground_truth(current_time - start_time)
        
        # ثبت داده‌ها
        collector.log_record(current_time, frame_idx, hr, hr_conf, rr, rr_conf, gt_hr, gt_rr)
        
        # نمایش اطلاعات روی تصویر وب‌کم
        status_text = f"HR: {hr:.1f} BPM (Conf: {hr_conf:.2f})" if hr_conf > 0.4 else "HR: Calculating..."
        rr_text = f"RR: {rr:.1f} BPM (Conf: {rr_conf:.2f})" if rr_conf > 0.3 else "RR: Calculating..."
        
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, rr_text, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to Quit & Analyze", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Real-time rPPG & Respiration Monitor", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_idx += 1
        
    cap.release()
    cv2.destroyAllWindows()
    
    # ذخیره و تحلیل نهایی داده‌ها پس از خروج
    collector.save_csv()
    collector.generate_statistics_and_plots()

if __name__ == "__main__":
    main()
