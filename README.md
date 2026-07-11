
#  README.md (English Version)

#  Advanced rPPG Medical Monitor 
### *Non-Contact Vital Signs Monitoring using FDA-Compliant Architecture*

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Framework](https://img.shields.io/badge/Flask-SocketIO-lightgrey?style=for-the-badge&logo=flask)
![AI](https://img.shields.io/badge/PyTorch-EfficientNet-orange?style=for-the-badge&logo=pytorch)
![Signal](https://img.shields.io/badge/Signal-Processing-green?style=for-the-badge&logo=scipy)

##  Overview
This repository contains a high-fidelity **Remote Photoplethysmography (rPPG)** monitoring system. It transforms a standard RGB camera into a medical-grade sensor capable of estimating **Heart Rate, SpO2, Respiratory Rate, and even reconstructed ECG signals** in real-time.

## 🔬 Core Technologies & Algorithms
This project implements several advanced methodologies often required for **FDA Class II** medical software (SaMD):

### 1. Signal Extraction & Enhancement
*   **PRISM (Principal Component Analysis Based rPPG):** Uses PCA to isolate the Blood Volume Pulse (BVP) from motion and lighting noise.
*   **Adaptive Kalman Filtering:** A dynamic noise reduction system that adjusts its process noise based on signal innovation, ensuring stable SpO2 and HR readings.
*   **NLMS Filter (Normalized Least Mean Squares):** Used in the `PWASpO2Processor` to cancel motion artifacts by using the Green channel as a noise reference for the Blue/Red channels.

### 2. AI-Driven Environmental Context
*   **Fitzpatrick Skin Type Classifier:** Uses an **EfficientNet-B0** deep learning model to categorize skin tones (Type I-VI), allowing the algorithm to adjust its gain and SpO2 coefficients accordingly.
*   **Neural Light Estimator:** A custom MLP (Multi-Layer Perceptron) that analyzes histogram features and facial symmetry to detect suboptimal lighting conditions.

### 3. Medical Innovations
*   **Synthetic ECG Reconstruction:** One of the most unique features—uses **APG (Acceleration Plethysmogram)** and second-order derivatives to reconstruct a synthetic ECG wave from the Green rPPG signal.
*   **Clinical Alarm System:** A logic-based engine that monitors vitals against clinical thresholds (Hypoxemia, Bradycardia, Tachycardia) with varying severity levels.

## 🛠 Tech Stack
*   **Computer Vision:** MediaPipe Face Mesh (468 Landmarks).
*   **Deep Learning:** PyTorch (Inference), Torchvision.
*   **Signal Processing:** SciPy (Butterworth Filters, Peak Detection), NumPy.
*   **Security:** Fernet Encryption (PII protection), SHA-256 Hashing for Patient ID.
*   **Database:** SQLite3 with WAL mode for thread-safe medical logging.

---

#  راهنمای پروژه (Farsi Version)

#  مانیتورینگ فوق پیشرفته علائم حیاتی (rPPG)
### *پایش غیرتماسی با معماری منطبق بر استانداردهای تجهیزات پزشکی*

##  معرفی پروژه
این پروژه یک سیستم **rPPG (فتوپلتیسموگرافی از راه دور)** بسیار دقیق است که دوربین معمولی لپ‌تاپ را به یک حسگر پزشکی تبدیل می‌کند. این سامانه قادر است **ضربان قلب، سطح اکسیژن خون (SpO2)، نرخ تنفس و سیگنال بازسازی شده ECG** را به صورت لحظه‌ای استخراج کند.

##  نوآوری‌ها و الگوریتم‌های کلیدی
در این پروژه از متدهای پیشرفته‌ای استفاده شده که در استانداردهای **FDA Class II** برای نرم‌افزارهای پزشکی (SaMD) حائز اهمیت هستند:

### ۱. استخراج و بهینه‌سازی سیگنال
*   **الگوریتم PRISM:** استفاده از تحلیل مولفه‌های اصلی (PCA) برای جداسازی پالس حجم خون از نویزهای محیطی و حرکتی.
*   **فیلتر کالمن تطبیقی (Adaptive Kalman):** یک سیستم هوشمند کاهش نویز که خود را با تغییرات لحظه‌ای سیگنال وفق داده و پایداری اعداد SpO2 و HR را تضمین می‌کند.
*   **فیلتر NLMS:** برای حذف آرتیفکت‌های حرکتی با استفاده از کانال سبز به عنوان مرجع نویز.

### ۲. هوش مصنوعی و تحلیل محیطی
*   **طبقه بندی پوست فیتزپاتریک:** استفاده از مدل **EfficientNet-B0** برای تشخیص تیپ پوستی (Type I-VI) جهت کالیبراسیون خودکار ضرایب اکسیژن خون.
*   **تخمین‌گر عصبی نور:** یک شبکه عصبی MLP برای تحلیل هیستوگرام و تقارن نوری چهره جهت هشدار در شرایط نوری نامناسب.

### ۳. ویژگی‌های منحصربه‌فرد پزشکی
*   **بازسازی سیگنال ECG:** استخراج موج شبیه سازی شده ECG از روی rPPG با استفاده از مشتق‌گیری مرتبه دوم (APG) و نگاشت غیرخطی.
*   **سیستم هشدار بالینی:** موتور منطقی برای بررسی علائم حیاتی بر اساس آستانه‌های پزشکی (افت اکسیژن، برادیکاردی، تاکی‌کاردی).

## 🛠 لایه‌های تکنولوژی
*   **بینایی ماشین:** MediaPipe Face Mesh (۴۶۸ نقطه کلیدی چهره).
*   **یادگیری عمیق:** PyTorch & Torchvision.
*   **پردازش سیگنال:** SciPy (فیلترهای باتروورث، تشخیص قله)، NumPy.
*   **امنیت داده:** رمزنگاری Fernet برای داده‌های حساس و هشینگ SHA-256 برای هویت بیماران.
*   **دیتا‌بیس:** SQLite3 با حالت WAL برای ثبت امن و همزمان لاگ‌های پزشکی.

---

###  Comparison Table (Clinical Context)

| Feature | This Project (rPPG) | Traditional Pulse Oximeter |
| :--- | :--- | :--- |
| **Contact** | Non-Contact (Remote) | Physical Contact Required |
| **Signal Processing** | PRISM + Adaptive Kalman | Basic Peak Detection |
| **Skin Tone Compensation** | EfficientNet-B0 AI | Usually None |
| **ECG Simulation** | Reconstructed from PPG | Not Available |
| **Motion Handling** | NLMS + Optical Flow | Very Sensitive |


