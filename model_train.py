# model_train.py
"""
Train TL models and save them in ./models/
Usage:
  python model_train.py --n-samples 5000
"""
import os
import argparse
import numpy as np
import pandas as pd
from utils import complex_gamma, characteristic_impedance, input_impedance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib

np.random.seed(42)

def sample_parameters(n):
    out = []
    for _ in range(n):
        # realistic narrowed ranges (adjust for your lab)
        Rp = 10 ** np.random.uniform(-5, -2)        # ohm/m
        Lp = 10 ** np.random.uniform(-8, -6)        # H/m
        Gp = 10 ** np.random.uniform(-10, -7)       # S/m
        Cp = 10 ** np.random.uniform(-12, -10)      # F/m
        length = np.random.uniform(0.5, 50.0)       # m
        freq = 10 ** np.random.uniform(4, 7)        # 10 kHz to 10 MHz
        ZL_real = np.random.uniform(5.0, 200.0)
        ZL_imag = np.random.uniform(-100.0, 100.0)
        ZL = ZL_real + 1j * ZL_imag
        out.append((Rp, Lp, Gp, Cp, length, freq, ZL))
    return out

def generate_dataset(n_samples=1000):
    rows = []
    samples = sample_parameters(n_samples)
    for Rp, Lp, Gp, Cp, length, freq, ZL in samples:
        gamma = complex_gamma(Rp, Lp, Gp, Cp, freq)
        Z0 = characteristic_impedance(Rp, Lp, Gp, Cp, freq)
        Zin = input_impedance(Z0, ZL, gamma, length)
        magZin = np.abs(Zin)
        phaseZin = np.angle(Zin)
        rows.append({
            'Rp': Rp, 'Lp': Lp, 'Gp': Gp, 'Cp': Cp,
            'length': length, 'freq': freq,
            'ZL_real': float(np.real(ZL)), 'ZL_imag': float(np.imag(ZL)),
            'Z0_real': float(np.real(Z0)), 'Z0_imag': float(np.imag(Z0)),
            'Zin_real': float(np.real(Zin)), 'Zin_imag': float(np.imag(Zin)),
            'magZin': float(magZin), 'phaseZin': float(phaseZin),
            'alpha': float(np.real(gamma)), 'beta': float(np.imag(gamma))
        })
    return pd.DataFrame(rows)

def prepare_features_targets(df):
    X = df[['Rp','Lp','Gp','Cp','length','freq','ZL_real','ZL_imag']].copy()
    X['log_Rp'] = np.log10(X['Rp'])
    X['log_Lp'] = np.log10(X['Lp'])
    X['log_Gp'] = np.log10(X['Gp'])
    X['log_Cp'] = np.log10(X['Cp'])
    X['log_freq'] = np.log10(X['freq'])
    X['log_length'] = np.log10(X['length'])
    Z0c = df['Z0_real'].values + 1j * df['Z0_imag'].values
    X['Z0_mag'] = np.abs(Z0c)
    X['Z0_phase'] = np.angle(Z0c)

    # Targets (transformed)
    y_logmag = np.log10(df['magZin'].values + 1e-12)
    y_phase = df['phaseZin'].values
    y_logalpha = np.log10(np.abs(df['alpha'].values) + 1e-12)
    y_beta = df['beta'].values

    Y = pd.DataFrame({
        'logmag': y_logmag,
        'phase': y_phase,
        'logalpha': y_logalpha,
        'beta': y_beta
    })
    return X, Y

def train_and_save_models(X_train, y_train, X_val, y_val, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    models = {}
    results = {}
    # Models: RF for log-like; GBR for phase/beta
    configs = {
        'logmag': ('rf', RandomForestRegressor(n_estimators=400, max_depth=20, random_state=42, n_jobs=-1)),
        'logalpha': ('rf', RandomForestRegressor(n_estimators=400, max_depth=20, random_state=42, n_jobs=-1)),
        'phase': ('gbr', GradientBoostingRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, random_state=42)),
        'beta': ('gbr', GradientBoostingRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, random_state=42))
    }

    for target, (kind, estimator) in configs.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('reg', estimator)])
        print(f"Training {target} using {kind} ...")
        pipe.fit(X_train, y_train[target].values)
        ypred = pipe.predict(X_val)
        r2 = r2_score(y_val[target].values, ypred)
        models[target] = pipe
        results[target] = float(r2)
        joblib.dump(pipe, os.path.join(out_dir, f"model_{target}.pkl"))
        print(f"Saved model_{target}.pkl | validation R2 = {r2:.4f}")

    # summary
    joblib.dump(results, os.path.join(out_dir, 'summary_r2.pkl'))
    return models, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=1000, help='How many dataset samples to generate')
    parser.add_argument('--out', type=str, default='models', help='Output directory for models/dataset')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("Generating dataset...")
    df = generate_dataset(n_samples=args.n_samples)
    df.to_csv(os.path.join(args.out, 'dataset.csv'), index=False)
    print("Saved dataset to", os.path.join(args.out, 'dataset.csv'))

    X, Y = prepare_features_targets(df)
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("Training models (this may take some minutes)...")
    models, results = train_and_save_models(X_train, y_train, X_val, y_val, args.out)
    print("Training complete. Summary R2:", results)
    print("Models and dataset are saved to:", args.out)
