import os
import logging
import io
import base64
import uuid
import numpy as np
import pandas as pd
import pytesseract
import cv2
import torch
import requests
import matplotlib.pyplot as plt
import statsmodels.api as sm
import face_recognition
import nltk
import spacy
from flask import Flask, request, jsonify, Response
from PIL import Image, ImageChops
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, IsolationForest
)
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Conv1D, BatchNormalization,
                                   MaxPooling1D, Dropout, Bidirectional, Input, Add)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from lifelines import KaplanMeierFitter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import pdfplumber
from statsmodels.tsa.arima.model import ARIMA

# Initialize NLP tools
nlp = spacy.load("en_core_web_sm")
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Set up Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Utility Functions --------------------
def allowed_file(filename, allowed_set=None):
    if allowed_set is None:
        allowed_set = {'png', 'jpg', 'jpeg', 'pdf', 'csv', 'xlsx', 'json'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

# -------------------- Data Processor --------------------
class DataProcessor:
    @staticmethod
    def import_data(file_path, sql_conn=None):
        if not os.path.exists(file_path) and not file_path.endswith('.sql'):
            raise FileNotFoundError(f"File {file_path} not found")

        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        elif file_path.endswith('.sql'):
            if not sql_conn:
                raise ValueError("SQL connection required")
            return pd.read_sql(file_path, sql_conn)
        else:
            raise ValueError("Unsupported file format")

    @staticmethod
    def clean_data(data):
        num_cols = data.select_dtypes(include=np.number).columns
        cat_cols = data.select_dtypes(exclude=np.number).columns

        num_imp = SimpleImputer(strategy='mean')
        cat_imp = SimpleImputer(strategy='most_frequent')

        data[num_cols] = num_imp.fit_transform(data[num_cols])
        data[cat_cols] = cat_imp.fit_transform(data[cat_cols])
        return data

    @staticmethod
    def preprocess_data(data):
        num_cols = data.select_dtypes(include=np.number).columns
        cat_cols = data.select_dtypes(exclude=np.number).columns

        scaler = StandardScaler()
        encoder = LabelEncoder()

        data[num_cols] = scaler.fit_transform(data[num_cols])
        for col in cat_cols:
            data[col] = encoder.fit_transform(data[col].astype(str))
        return data

# -------------------- Digital Lending --------------------
class DigitalLending:
    @staticmethod
    def credit_scoring(x_train, x_test, y_train, y_test):
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100),
            "XGBoost": XGBClassifier()
        }

        results = {}
        for name, model in models.items():
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            results.update({
                f"{name}_Accuracy": accuracy_score(y_test, y_pred),
                f"{name}_F1": f1_score(y_test, y_pred),
                f"{name}_ROC_AUC": roc_auc_score(y_test, model.predict_proba(x_test)[:,1])
            })
        return results

    @staticmethod
    def customer_segmentation(data, n_clusters=5):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        kmeans = KMeans(n_clusters=n_clusters)
        data['Cluster'] = kmeans.fit_predict(scaled)

        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled)
        return {
            'clusters': data['Cluster'].tolist(),
            'pca_components': components.tolist()
        }

# -------------------- Customer Acquisition --------------------
class CustomerAcquisition:
    @staticmethod
    def lead_score(x_train, x_test, y_train, y_test):
        model = GradientBoostingClassifier()
        model.fit(x_train, y_train)
        return {
            "accuracy": accuracy_score(y_test, model.predict(x_test)),
            "feature_importance": dict(zip(x_train.columns, model.feature_importances_))
        }

    @staticmethod
    def churn_prediction(x_train, x_test, y_train, y_test, data, save_path='best_churn_model.h5', epochs=50, batch_size=32):
        input_shape = x_train.shape[1:]
        inputs = Input(shape=input_shape)

        # Model architecture
        x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)

        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(32))(x)
        x = Dropout(0.2)(x)

        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Callbacks
        callbacks = [
            EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
            ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss')
        ]

        # Training
        history = model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Predictions
        predictions = model.predict(x_test).flatten()
        pred_labels = (predictions > 0.5).astype(int)

        # Survival analysis
        kmf = KaplanMeierFitter()
        kmf.fit(durations=data['duration'], event_observed=data['churn'])

        plt.figure()
        kmf.plot_survival_function()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        plt.close()

        return {
            "accuracy": accuracy_score(y_test, pred_labels),
            "roc_auc": roc_auc_score(y_test, predictions),
            "survival_plot": f"data:image/png;base64,{plot_url}",
            "history": history.history
        }

    @staticmethod
    def predictive_analysis(data, target_column='sales', forecast_steps=12):
        model = ARIMA(data[target_column], order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_steps)

        plt.figure()
        plt.plot(data[target_column], label='Historical')
        plt.plot(forecast, label='Forecast')
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        plt.close()

        return {
            "forecast": forecast.tolist(),
            "plot": f"data:image/png;base64,{plot_url}"
        }

# -------------------- Employee & Vendor BGV --------------------
class EmployeeVendorBGV:
    @staticmethod
    def train_and_evaluate_model(X_train, X_test, y_train, y_test):
        models = {
            'SVM': SVC(kernel='linear', probability=True),
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier()
        }

        best_model = None
        highest_accuracy = 0
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for model_name, model in models.items():
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
            mean_accuracy = scores.mean()

            if mean_accuracy > highest_accuracy:
                highest_accuracy = mean_accuracy
                best_model = model

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return best_model, accuracy

    @staticmethod
    def analyze_document(text):
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        sentiment = sia.polarity_scores(text)
        return {
            "entities": entities,
            "sentiment": sentiment,
            "risk_score": (sentiment['compound'] + 1) * 50
        }

    @staticmethod
    def validate_document(text):
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform([text])
        model = LogisticRegression()
        model.fit(X, np.random.randint(2, size=1))
        return model.predict_proba(X)[0][1]

    @staticmethod
    def risk_scoring_with_ensemble(X_train, X_test, y_train, y_test):
        ensemble_model = VotingClassifier(estimators=[
            ('svm', SVC(kernel='linear', probability=True)),
            ('nb', MultinomialNB()),
            ('log_reg', LogisticRegression(max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', XGBClassifier())
        ], voting='soft')

        ensemble_model.fit(X_train, y_train)
        y_pred = ensemble_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return ensemble_model, accuracy

# --------------------- Document Forgery Analysis -------------------------------------
def perform_ela_analysis(image_path, quality=90):
    temp_filename = f"temp_ela_{uuid.uuid4()}.jpg"
    try:
        original = Image.open(image_path).convert('RGB')
        original.save(temp_filename, 'JPEG', quality=quality, optimize=True)
        compressed = Image.open(temp_filename)

        ela_image = ImageChops.difference(original, compressed)
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema]) or 1  # Prevent division by zero
        scale = 255.0 / max_diff
        ela_image = ela_image.point(lambda x: x * scale)

        buf = io.BytesIO()
        ela_image.save(buf, format='PNG', compress_level=0)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# -------------------- Fraud Detection --------------------
class FraudDetection:
    @staticmethod
    def residual_block(x, filters, kernel_size, block_name):
        shortcut = x
        x = Conv1D(filters, kernel_size, padding='same', activation='relu', name=f'{block_name}_conv1')(x)
        x = BatchNormalization(name=f'{block_name}_bn1')(x)
        x = Conv1D(filters, kernel_size, padding='same', activation='relu', name=f'{block_name}_conv2')(x)
        x = BatchNormalization(name=f'{block_name}_bn2')(x)
        x = Add(name=f'{block_name}_add')([x, shortcut])
        return x

    @staticmethod
    def behavior_analysis(sequences, labels=None, epochs=50, batch_size=32, validation_split=0.2, save_path='best_fraud_model.h5'):
        input_shape = sequences.shape[1:]
        inputs = Input(shape=input_shape)

        # Initial Conv Layer
        x = Conv1D(64, 3, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.3)(x)

        # Residual Blocks
        x = FraudDetection.residual_block(x, 64, 3, 'res1')
        x = MaxPooling1D(2)(x)
        x = FraudDetection.residual_block(x, 64, 3, 'res2')

        # BiLSTM Layers
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(32))(x)
        x = Dropout(0.3)(x)

        # Output Layer
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=1e-3),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        if labels is not None:
            callbacks = [
                EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
                ModelCheckpoint(filepath=save_path, save_best_only=True, monitor='val_loss', verbose=1)
            ]

            history = model.fit(
                sequences, labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )

            predictions = model.predict(sequences).flatten()
            pred_labels = (predictions > 0.5).astype(int)

            return {
                "accuracy": accuracy_score(labels, pred_labels),
                "roc_auc": roc_auc_score(labels, predictions),
                "predictions": predictions.tolist(),
                "history": history.history,
                "model_saved": os.path.exists(save_path)
            }
        else:
            predictions = model.predict(sequences).flatten()
            return {
                "predictions": predictions.tolist(),
                "note": "No labels provided. Inference mode only."
            }

    @staticmethod
    def detect_anomalies(data):
        clf = IsolationForest()
        scores = clf.fit_predict(data)
        return {
            "anomaly_scores": (-clf.score_samples(data)).tolist(),
            "threshold": np.percentile(scores, 95)
        }

# -------------------- Digital Onboarding --------------------
class DigitalOnboarding:
    def __init__(self):
        self.known_encoding = None
        self.video_capture = None
        self.running = False
        self.frame_lock = threading.Lock()
        self.current_frame = None

    def load_reference_face(self, image_path):
        try:
            image = face_recognition.load_image_file(image_path)
            self.known_encoding = face_recognition.face_encodings(image)[0]
        except Exception as e:
            logger.error(f"Error loading reference face: {str(e)}")

    def start_capture(self):
        self.video_capture = cv2.VideoCapture(0)
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            processed_frame = self._preprocess_image(frame)
            text = self._process_ocr(processed_frame)
            frame = self._process_faces(frame)

            with self.frame_lock:
                self.current_frame = frame

    def _preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        return cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def _process_ocr(self, image):
        custom_config = r'--oem 3 --psm 6'
        return pytesseract.image_to_string(image, config=custom_config)

    def _process_faces(self, frame):
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            if self.known_encoding:
                match = face_recognition.compare_faces([self.known_encoding], face_encoding)[0]
                color = (0, 255, 0) if match else (0, 0, 255)
                label = "Verified" if match else "Not Verified"
            else:
                color = (255, 0, 0)
                label = "No Reference"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    def get_frame(self):
        with self.frame_lock:
            if self.current_frame is None:
                return None
            _, jpeg = cv2.imencode('.jpg', self.current_frame)
            return jpeg.tobytes()

    def stop_capture(self):
        self.running = False
        if self.video_capture:
            self.video_capture.release()

# -------------------- Claim Automation --------------------
class ClaimAutomation:
    class CNNModel(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 16 * 16, 512),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    @staticmethod
    def process_ocr(image_path):
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            _, thresholded = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            custom_config = r'--oem 3 --psm 6'
            return pytesseract.image_to_string(thresholded, config=custom_config).strip()
        except Exception as e:
            logger.error(f"OCR Error: {str(e)}")
            return ""

    @staticmethod
    def train_cnn(x_train, y_train, x_test, y_test, epochs=10):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = [(transform(Image.fromarray(img)), lbl) for img, lbl in zip(x_train, y_train)]
        test_dataset = [(transform(Image.fromarray(img)), lbl) for img, lbl in zip(x_test, y_test)]

        model = ClaimAutomation.CNNModel().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            model.train()
            for inputs, labels in DataLoader(train_dataset, batch_size=32):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return model

    @staticmethod
    def train_random_forest(x_train, y_train, x_test, y_test):
        model = RandomForestClassifier(n_estimators=200)
        scores = cross_val_score(model, x_train, y_train, cv=5)
        model.fit(x_train, y_train)
        return {
            "cv_scores": scores.tolist(),
            "test_accuracy": accuracy_score(y_test, model.predict(x_test))
        }

# Initialize modules
onboarding_system = DigitalOnboarding()
if os.path.exists("user.jpg"):
    onboarding_system.load_reference_face("user.jpg")

# -------------------- Flask Routes --------------------
@app.route('/process', methods=['POST'])
def handle_processing():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        if file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
            text = ClaimAutomation.process_ocr(file_path)
            analysis = EmployeeVendorBGV.analyze_document(text)
            return jsonify(analysis)

        elif file.filename.lower().endswith(('csv', 'xlsx', 'json')):
            df = DataProcessor.import_data(file_path)
            cleaned = DataProcessor.clean_data(df)
            processed = DataProcessor.preprocess_data(cleaned)

            target = request.form.get('target')
            if not target:
                return jsonify({"error": "Target column is required"}), 400

            X_train, X_test, y_train, y_test = train_test_split(
                processed.drop(columns=[target]), processed[target], test_size=0.2)

            task = request.form.get('task')
            if task == 'lending':
                results = DigitalLending.credit_scoring(X_train, X_test, y_train, y_test)
            elif task == 'acquisition':
                results = CustomerAcquisition.lead_score(X_train, X_test, y_train, y_test)
            elif task == 'fraud':
                results = FraudDetection.detect_anomalies(processed)
            else:
                return jsonify({"error": "Invalid task"}), 400

            return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/onboarding/start', methods=['POST'])
def start_onboarding():
    onboarding_system.start_capture()
    return jsonify({"status": "started"})

@app.route('/onboarding/stream')
def onboarding_stream():
    def generate():
        while True:
            frame = onboarding_system.get_frame()
            if frame is None:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/onboarding/stop', methods=['POST'])
def stop_onboarding():
    onboarding_system.stop_capture()
    return jsonify({"status": "stopped"})

@app.route('/doc/forgery', methods=['POST'])
def check_document_forgery():
    file = request.files.get('file')
    if not file or not allowed_file(file.filename, {'jpg', 'jpeg', 'png'}):
        return jsonify({"error": "Image file required"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    ela_result = perform_ela_analysis(file_path)

    return jsonify({
        "ela_analysis": f"data:image/png;base64,{ela_result}",
        "note": "Bright areas may indicate potential manipulation"
    })

@app.route('/claim/ocr', methods=['POST'])
def claim_ocr():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    text = ClaimAutomation.process_ocr(file_path)
    return jsonify({"text": text})

@app.route('/claim/train', methods=['POST'])
def train_claim_models():
    data = request.get_json()
    cnn_model = ClaimAutomation.train_cnn(
        np.array(data['x_train']),
        np.array(data['y_train']),
        np.array(data['x_test']),
        np.array(data['y_test'])
    )
    rf_results = ClaimAutomation.train_random_forest(
        np.array(data['x_train']),
        np.array(data['y_train']),
        np.array(data['x_test']),
        np.array(data['y_test'])
    )
    return jsonify({
        "cnn_status": "trained",
        "rf_results": rf_results
    })

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    if os.name == 'nt':
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    app.run(host='0.0.0.0', port=5000, threaded=True)
