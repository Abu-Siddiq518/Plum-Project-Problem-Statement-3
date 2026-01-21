# 🧪 Plum Project – Problem Statement 3  
## Medical Report Simplifier 🩺

A Flask-based web application that simplifies medical reports by analyzing **text** and **image-based medical reports** using OCR and reference medical databases. The system extracts medical values, normalizes them, and provides easy-to-understand explanations for patients and users.

---

## 🚀 Features

- 📄 **Text-based medical report analysis**
- 🖼️ **Image-based medical report analysis (OCR using Tesseract)**
- 📊 Medical value normalization using reference databases
- 💡 Simple explanations for medical terms
- 🌐 Web-based user interface
- 🔌 REST API support

---

## 📦 Prerequisites

Make sure the following are installed on your system:

- **Python 3.8 or higher**
- **Tesseract OCR** (required for image processing)

---

## 🧩 Installation Guide

### 🔹 Step 1: Install Tesseract OCR (Windows)

1. Download the installer from:  
   https://github.com/UB-Mannheim/tesseract/wiki

2. Install Tesseract  
   **Recommended path:**

3. Add Tesseract to system PATH  
**OR** update the path manually in `app.py`:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```
🔹 Step 2: Clone the Repository
git clone https://github.com/Abu-Siddiq518/Plum-Project-Problem-Statement-3.git
cd medical-report-simplifier

🔹 Step 3: Create and Activate Virtual Environment
# Create virtual environment
python -m venv venv

Activate the environment

Windows
venv\Scripts\activate

Linux / macOS
source venv/bin/activate

🔹 Step 4: Install Dependencies
pip install -r requirements.txt

▶️ Run the Application
python app.py

🌐 Access the Web Interface

Open your browser and visit:http://localhost:5000


| Method | Endpoint             | Description                   |
| ------ | -------------------- | ----------------------------- |
| GET    | `/`                  | Web interface                 |
| GET    | `/api/health`        | Health check                  |
| POST   | `/api/analyze/text`  | Analyze text medical reports  |
| POST   | `/api/analyze/image` | Analyze image medical reports |



