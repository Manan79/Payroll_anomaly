
# 🧠 Payroll Anomaly Detection App

A Streamlit-based web app that identifies anomalies in payroll data using machine learning and visualizes results with interactive charts.

---

## 🚀 Features

- ✅ Upload and analyze payroll data (CSV)
- 🧪 Detect anomalies using a trained ML model (autoencoder-based)
- 📊 Visualize anomaly scores and feature importance
- 🧬 Explain why each data point is anomalous
- 📈 Interactive plots powered by Plotly
- 💾 Downloadable results with anomaly flags

---

## 📂 Project Structure

```bash
📦 payroll-anomaly-detection/
├── streamlit_app.py          # Main Streamlit app
├── model/                    # Trained ML model (joblib/h5)
├── utils/                    # Preprocessing & helper functions
├── data/                     # Example datasets
├── requirements.txt          # Python dependencies
└── runtime.txt               # Python version pinning (for Streamlit Cloud)
```

---

## 📦 Setup Instructions

### ✅ 1. Clone this repo
```bash
git clone https://github.com/your-username/payroll-anomaly-detection.git
cd payroll-anomaly-detection
```

### ✅ 2. Create virtual environment

> **You must use Python 3.10 or 3.11**
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### ✅ 3. Install dependencies
```bash
pip install -r requirements.txt
```

> Optional: Update pip if prompted  
```bash
pip install --upgrade pip
```

### ✅ 4. Run the app
```bash
streamlit run streamlit_app.py
```

---

## 🛠 Dependencies

Your `requirements.txt` should include:

```txt
matplotlib
scikit-learn
seaborn
pandas
numpy
plotly
tensorflow-cpu==2.11.0
joblib
openpyxl
streamlit
```

> ⚠️ Do not use Python 3.12 or higher – TensorFlow is incompatible!

Your `runtime.txt` (for Streamlit Cloud):
```txt
python-3.10
```

---

## 📊 Visualizations Included

- **Radar Plot**: Feature impact in anomalies
- **Violin Plot**: Distribution of anomaly scores
- **Bar Charts**: Feature-wise value distribution
- **Anomaly Explanations**: Highlight which features deviate from average

---

## 📁 Sample Data

You can test the app with any structured payroll-like CSV, with numerical features (e.g., hours, salary, bonuses). Example:

```csv
EmployeeID,HoursWorked,BasePay,Bonus,Overtime
101,160,5000,300,20
...
```

---

## 👨‍💻 Authors

Built by Mannan Sood and Achuki Saini  
🌐 https://www.linkedin.com/in/mannan-sood-a38688253/

---

## 📜 License
Feel free to use, share, and modify with attribution.
