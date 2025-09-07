⚡ Transmission Line Predictor

A web-based interactive tool (built with Streamlit) for analyzing transmission line parameters, visualizing standing wave patterns, and predicting input impedance using Machine Learning models.

🚀 Features

✅ Input distributed parameters , line length, frequency, and load impedance
✅ Compute:

Characteristic Impedance 

Propagation Constant 

Input Impedance 

Reflection Coefficient & VSWR

Wavelength 


✅ Standing Wave Visualization (voltage distribution along the line)
✅ Machine Learning Prediction with 99%+ accuracy (RandomForest + GradientBoosting)
✅ Beautiful dark theme UI with gradient background and styled sidebar
✅ Ready for deployment on Streamlit Cloud / Render / Docker


---

📂 Project Structure

transmission-line-app/
│── app.py                # Streamlit main app
│── model_train.py        # Training script for ML models
│── utils.py              # Transmission line equations
│── requirements.txt      # Dependencies
│── .gitignore            # Ignore unnecessary files
│── .streamlit/
│     └── config.toml     # Theme config
│── models/               # Saved trained models (generated after training)


---

⚙️ Installation

1. Clone Repository

git clone https://github.com/YOURUSERNAME/transmission-line-app.git
cd transmission-line-app

2. Create Virtual Environment (optional but recommended)

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Install Requirements

pip install -r requirements.txt


---

🧠 Train the ML Model

Before running the app, train models with at least 1000–5000 samples:

python model_train.py --n 5000

This generates:

models/model_magZin.pkl

models/model_phase.pkl

models/model_alpha.pkl

models/model_beta.pkl

models/dataset.csv



---

▶️ Run the Streamlit App

streamlit run app.py

Access in browser at: http://localhost:8501


---

🌐 Deployment

Deploy to Streamlit Cloud

1. Push this repo to GitHub.


2. Go to Streamlit Cloud, select repo, and deploy.



Deploy to Render (optional)

Add a Dockerfile or render.yaml and deploy via Render Dashboard.

📚 Theory

This app implements classical Transmission Line Theory from Electromagnetics:

Telegrapher’s Equations

Propagation Constant

Characteristic Impedance

Standing Wave Ratio (VSWR)

Reflection Coefficient


Machine learning models are trained to predict input impedance and wave parameters for fast evaluation.


