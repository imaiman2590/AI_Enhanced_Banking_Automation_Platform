

---

# 🧠 Intelligent Financial Automation Platform (IFAP)

A comprehensive AI/ML-driven platform built with Flask for automating key functions in digital lending, fraud detection, document verification, onboarding, and claims processing in financial services.

---

## 🚀 Features

### ✅ Digital Lending

* Credit scoring using Logistic Regression, Random Forest, and XGBoost.
* Customer segmentation with KMeans + PCA visualization.

### 📈 Customer Acquisition

* Lead scoring with Gradient Boosting.
* Churn prediction using deep learning (Conv1D + BiLSTM).
* Predictive sales analysis using ARIMA forecasting.

### 🧾 Employee & Vendor Background Verification (BGV)

* Named Entity Recognition (NER) + Sentiment Analysis.
* Risk scoring using ensemble machine learning models.
* Document classification & forgery detection (OCR + ELA).

### 🕵️‍♀️ Fraud Detection

* Behavioral analysis using a Residual CNN + BiLSTM model.
* Anomaly detection with Isolation Forest.

### 👤 Digital Onboarding

* Live face verification via webcam using `face_recognition`.
* Real-time OCR on video feed.

### 📑 Claim Automation

* OCR-based document analysis using OpenCV + Tesseract.
* CNN-based image classification (PyTorch).
* Random Forest for structured claims processing.

---

## 🛠️ Tech Stack

* **Frameworks**: Flask, PyTorch, TensorFlow, scikit-learn
* **NLP**: spaCy, NLTK, VADER
* **CV/OCR**: OpenCV, Tesseract, face\_recognition
* **Visualization**: Matplotlib
* **Data Handling**: pandas, numpy, pdfplumber
* **Forecasting**: statsmodels (ARIMA), lifelines (Kaplan-Meier)

---

## 📂 Project Structure

```bash
├── app.py                   # Main Flask application
├── uploads/                # Uploaded files directory
├── requirements.txt        # Dependencies
└── README.md               # This file
```

---

## 🔧 Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/your-repo/ifap.git
cd ifap
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **(Windows Only) Configure Tesseract**

```python
# In app.py, set the path if not in environment
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

4. **Run the App**

```bash
python app.py
```

Navigate to: `http://localhost:5000`

---

## 📡 API Endpoints

### `/process` `POST`

* Processes CSV/Excel/JSON/image files.
* Tasks: `lending`, `acquisition`, `fraud`.

### `/onboarding/start` `POST`

* Starts webcam-based onboarding.

### `/onboarding/stream` `GET`

* Streams live webcam video with face recognition overlay.

### `/onboarding/stop` `POST`

* Stops video capture.

### `/doc/forgery` `POST`

* Upload image to perform Error Level Analysis (ELA).

### `/claim/ocr` `POST`

* Extracts text from uploaded document image.

### `/claim/train` `POST`

* Trains both CNN and Random Forest models on uploaded image data.

---

## 📦 Example Usage (cURL)

```bash
curl -X POST -F "file=@example.csv" -F "target=label" -F "task=lending" http://localhost:5000/process
```

---

## ✅ To-Do / Improvements

* Add authentication for endpoints.
* Export trained models.
* Build interactive UI (React or Streamlit).
* Add Docker support.

---

## 🧠 Contributors

Maintained by your organization/team. Feel free to open issues or contribute to extend functionality.

---

## ⚖️ License

This project is licensed under the MIT License.

---
