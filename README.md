# CardioGuard AI - Heart Disease Deep Learning Prediction System

An end-to-end clinical decision support application powered by a **Deep Residual Neural Network (Tabular ResNet)** in TensorFlow / Keras, built on top of [neural-network-tensorflow-auc-0-9529.ipynb](file:///c:/Users/hp/OneDrive/文档/GitHub/dl2/neural-network-tensorflow-auc-0-9529.ipynb).

---

## 🌟 Key Highlights & Performance Metrics

| Metric | Score / Value | Description |
| :--- | :--- | :--- |
| **ROC-AUC Score** | **0.9529 (95.29%)** ⭐ | Superior discrimination capability between healthy and heart disease patients |
| **Test Accuracy** | **88.48%** | Evaluated across 126,000 unseen test patient records |
| **Precision** | **87.51%** | Minimizes false positive diagnoses |
| **Recall (Sensitivity)** | **86.66%** | High sensitivity for detecting true cardiovascular risk |
| **F1-Score** | **87.08%** | Balanced harmonic mean |
| **Dataset Size** | **630,000 records** | Trained on Kaggle Playground Series s6e2 Heart Disease data |
| **Inference Latency** | **< 1 ms / patient** | Suitable for real-time edge and clinical workflows |

---

## 🚀 How to Run the Application

### Option 1: Quick Launch via Windows Batch File
Double click or run:
```bat
run_app.bat
```

### Option 2: Direct Command Line
Using the isolated virtual environment:
```powershell
.venv\Scripts\streamlit.exe run app.py
```

The application will launch automatically at `http://localhost:8501`.

---

## 🩺 Application Features

### 1. 🔮 Interactive Individual Patient Risk Assessment
- **Clinical Archetype Presets**: Load pre-configured clinical profiles with 1 click:
  - *Low Risk / Athletic Profile*
  - *Borderline / Moderate Profile*
  - *High Risk / Angina Profile*
  - *Reset to Default*
- **Categorized Inputs**:
  - Demographics & Vitals (`Age`, `Sex`, `BP`, `Cholesterol`)
  - Symptoms & Chemistry (`Chest pain type`, `FBS over 120`, `EKG results`, `Number of vessels fluro`)
  - Stress Diagnostics (`Max HR`, `Exercise angina`, `ST depression`, `Slope of ST`, `Thallium`)
- **Real-Time Prediction Engine**:
  - Continuous risk probability score ($0.0\%$ to $100.0\%$)
  - Dynamic Plotly Gauge with customizable sensitivity cutoff (default: $0.50$)
  - Risk categorization tier: *Low*, *Moderate*, *High*, or *Critical*
  - Actionable clinical recommendations and automated text report generation with 1-click download

### 2. 📁 High-Throughput Batch Patient Screening
- Upload custom patient cohort CSVs for instant batch inference
- Built-in **Download Sample Patients CSV** button (`sample_patients_test.csv`)
- High-speed processing (>1,000 patients/sec)
- Cohort risk distribution histograms, risk-tier pie charts, and downloadable enriched CSV reports

### 3. 📊 Verified Model Performance & Analytics
- Interactive ROC Curve reproducing the notebook's benchmark (AUC = 0.9529)
- Interactive Confusion Matrix heatmap (62,584 TN, 6,980 FP, 7,530 FN, 48,906 TP)
- Training & validation loss/accuracy curves across epochs with Early Stopping visualization
- Key feature correlation chart (highlighting `Thallium`, `Chest pain type`, `Exercise angina`, etc.)

### 4. 🧠 Deep Neural Network Architecture Explorer
- Interactive inspection of the Tabular Residual Neural Network:
  - Input Layer $(13,)$ $\rightarrow$ Batch Normalization
  - Residual Block 1: `Dense(32) -> BN -> ReLU -> Dense(32) -> BN -> Dense(13) -> BN -> Add([x, out])`
  - Residual Block 2: `Dense(32) -> BN -> ReLU -> Dense(32) -> BN -> Dense(13) -> BN -> Add([x, out])`
  - Regularization: `Dropout(0.30)`
  - Output Head: `Dense(1, activation='sigmoid')`
- Real-time parameter counter and model summary viewer

---

## 📂 Repository File Structure

- [app.py](file:///c:/Users/hp/OneDrive/文档/GitHub/dl2/app.py): Complete end-to-end Streamlit web application
- [neural-network-tensorflow-auc-0-9529.ipynb](file:///c:/Users/hp/OneDrive/文档/GitHub/dl2/neural-network-tensorflow-auc-0-9529.ipynb): Original research and training notebook
- [best_heart_disease_model.keras](file:///c:/Users/hp/OneDrive/文档/GitHub/dl2/best_heart_disease_model.keras): Best checkpoint saved during training
- [heart_disease_model_final.keras](file:///c:/Users/hp/OneDrive/文档/GitHub/dl2/heart_disease_model_final.keras): Final epoch saved model
- [heart_disease_model_architecture.json](file:///c:/Users/hp/OneDrive/文档/GitHub/dl2/heart_disease_model_architecture.json): JSON architecture definition
- [heart_disease_model_weights.weights.h5](file:///c:/Users/hp/OneDrive/文档/GitHub/dl2/heart_disease_model_weights.weights.h5): Model weights file
- [sample_patients_test.csv](file:///c:/Users/hp/OneDrive/文档/GitHub/dl2/sample_patients_test.csv): Sample patient cohort for batch prediction demo
- [run_app.bat](file:///c:/Users/hp/OneDrive/文档/GitHub/dl2/run_app.bat): 1-click Windows launcher
