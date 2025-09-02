# app.py (beautified)
import os
import joblib
import matplotlib.pyplot as plt
from utils import standing_wave
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from utils import complex_gamma, characteristic_impedance, input_impedance, voltage_current_along_line, reflection_coefficient

st.set_page_config(
    page_title="Transmission Line Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_DIR = "models"

# Load ML models if present
def try_load_models():
    models = {}
    all_present = True
    targets = ['logmag','phase','logalpha','beta']
    for t in targets:
        p = os.path.join(MODEL_DIR, f"model_{t}.pkl")
        if os.path.exists(p):
            models[t] = joblib.load(p)
        else:
            all_present = False
            models[t] = None
    return models, all_present

models, models_present = try_load_models()

# Sidebar inputs
st.sidebar.title("⚙️ Transmission Line Parameters")
st.sidebar.markdown("Provide inputs below:")

with st.sidebar.expander("Line Characteristics"):
    Rp = st.number_input("R' (Ω/m)", value=1e-4, format="%.6g")
    Lp = st.number_input("L' (H/m)", value=1e-7, format="%.6g")
    Gp = st.number_input("G' (S/m)", value=1e-9, format="%.6g")
    Cp = st.number_input("C' (F/m)", value=1e-11, format="%.6g")
    length = st.number_input("Length (m)", value=5.0, format="%.4f")

with st.sidebar.expander("Signal & Load"):
    freq = st.number_input("Frequency (Hz)", value=1e6, format="%.6g")
    ZL_real = st.number_input("ZL Real (Ω)", value=50.0, format="%.4f")
    ZL_imag = st.number_input("ZL Imag (Ω)", value=0.0, format="%.4f")
ZL = complex(ZL_real, ZL_imag)

# Page Title
st.title("⚡ Transmission Line Analyzer")
st.caption("Compute analytic TL parameters, predict via ML, and visualize waveforms interactively.")

# Tabs for sections
tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🤖 ML Predictions", "📈 Waveforms"])

with tab1:
    st.subheader("Analytic Results")
    gamma = complex_gamma(Rp, Lp, Gp, Cp, freq)
    alpha = np.real(gamma)
    beta = np.imag(gamma)
    Z0 = characteristic_impedance(Rp, Lp, Gp, Cp, freq)
    Zin = input_impedance(Z0, ZL, gamma, length)
    GammaL = reflection_coefficient(Z0, ZL)
    VSWR = (1 + np.abs(GammaL)) / (1 - np.abs(GammaL) + 1e-12)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Characteristic |Z0| (Ω)", f"{np.abs(Z0):.3f}")
    col2.metric("Z0 Phase (°)", f"{np.degrees(np.angle(Z0)):.2f}")
    col3.metric("Alpha (Np/m)", f"{alpha:.3e}")
    col4.metric("Beta (rad/m)", f"{beta:.3e}")

    st.markdown("#### Input Impedance")
    st.write(f"Zin = {Zin.real:.3f} + j{Zin.imag:.3f} Ω")
    st.write(f"|Zin| = {np.abs(Zin):.3f} Ω, ∠Zin = {np.degrees(np.angle(Zin)):.2f} °")

    st.markdown("#### Reflection Coefficient")
    st.write(f"|Γ| = {np.abs(GammaL):.3f}, ∠Γ = {np.degrees(np.angle(GammaL)):.2f} °, VSWR = {VSWR:.3f}")

with tab2:
    st.subheader("Machine Learning Predictions")
    if models_present:
        # Build feature vector
        Xrow = {
            'Rp': Rp, 'Lp': Lp, 'Gp': Gp, 'Cp': Cp, 'length': length, 'freq': freq,
            'ZL_real': ZL.real, 'ZL_imag': ZL.imag,
            'log_Rp': np.log10(Rp), 'log_Lp': np.log10(Lp),
            'log_Gp': np.log10(Gp), 'log_Cp': np.log10(Cp),
            'log_freq': np.log10(freq), 'log_length': np.log10(length),
            'Z0_mag': np.abs(Z0), 'Z0_phase': np.angle(Z0)
        }
        Xdf = pd.DataFrame([Xrow])

        pred_logmag = models['logmag'].predict(Xdf)[0]
        pred_phase = models['phase'].predict(Xdf)[0]
        pred_logalpha = models['logalpha'].predict(Xdf)[0]
        pred_beta = models['beta'].predict(Xdf)[0]

        pred_mag = 10 ** pred_logmag
        pred_alpha = 10 ** pred_logalpha

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ML |Zin| (Ω)", f"{pred_mag:.3f}")
        c2.metric("ML ∠Zin (°)", f"{np.degrees(pred_phase):.2f}")
        c3.metric("ML Alpha (Np/m)", f"{pred_alpha:.3e}")
        c4.metric("ML Beta (rad/m)", f"{pred_beta:.3e}")

        st.markdown("#### Comparison Table")
        comp = {
            'Quantity': ['|Zin| (Ω)','Phase (°)','Alpha (Np/m)','Beta (rad/m)'],
            'Analytic': [f"{np.abs(Zin):.3f}", f"{np.degrees(np.angle(Zin)):.2f}", f"{alpha:.3e}", f"{beta:.3e}"],
            'ML Pred': [f"{pred_mag:.3f}", f"{np.degrees(pred_phase):.2f}", f"{pred_alpha:.3e}", f"{pred_beta:.3e}"]
        }
        st.table(pd.DataFrame(comp))
    else:
        st.warning("⚠️ ML models not found. Run `python model_train.py` to train them.")

with tab3:
    st.subheader("Interactive Waveforms")

    zs, Vz, Iz, GammaL = voltage_current_along_line(Z0, ZL, gamma, length, N=400)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=zs, y=np.abs(Vz), mode="lines", name="|V(z)|"))
    fig.add_trace(go.Scatter(x=zs, y=np.abs(Iz), mode="lines", name="|I(z)|"))
    fig.update_layout(title="Voltage & Current along the Line", xaxis_title="z (m)", yaxis_title="Magnitude")
    st.plotly_chart(fig, use_container_width=True)

    freqs = np.logspace(np.log10(max(1.0, freq/100)), np.log10(freq*100), 300)
    Gamma_mag = [np.abs(reflection_coefficient(characteristic_impedance(Rp,Lp,Gp,Cp,f), ZL)) for f in freqs]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=freqs, y=Gamma_mag, mode="lines", name="|Γ|"))
    fig2.update_layout(
        title="Reflection Coefficient vs Frequency",
        xaxis=dict(title="Frequency (Hz)", type="log"),
        yaxis=dict(title="|Γ|")
    )
    st.plotly_chart(fig2, use_container_width=True)

z_norm, Vz, Gamma = standing_wave(Rp, Lp, Gp, Cp, freq, ZL_real, ZL_imag)

st.subheader("Standing Wave Pattern (Voltage)")

fig, ax = plt.subplots()
ax.plot(z_norm, Vz, label=f"|V(z)|, Γ = {np.round(np.abs(Gamma),2)}")
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0", "λ/4", "λ/2", "3λ/4", "λ"])
ax.set_xlabel("Distance along line")
ax.set_ylabel("Voltage Magnitude")
ax.set_title("Voltage Standing Wave")
ax.grid(True, linestyle="--", alpha=0.7)
ax.legend()
st.pyplot(fig)



st.markdown("---")
st.caption("Made with ❤️ using Streamlit, Plotly, and Scikit-learn by Sagarrr❤️")
