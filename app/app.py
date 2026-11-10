import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn

CLASS_NAMES = ["N: normal", "S: supraventricular ectopic", "V: ventricular ectopic", "F: fusion", "Q: unknown / paced"]
FS = 125

class ECGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(128, 187) * math.sqrt(2.0 / 187))
        self.b1 = nn.Parameter(torch.zeros(128))
        self.W2 = nn.Parameter(torch.randn(64, 128) * math.sqrt(2.0 / 128))
        self.b2 = nn.Parameter(torch.zeros(64))
        self.W3 = nn.Parameter(torch.randn(5, 64) * math.sqrt(2.0 / 64))
        self.b3 = nn.Parameter(torch.zeros(5))

    def forward(self, x):
        a1 = torch.clamp(x @ self.W1.T + self.b1, min=0)
        a2 = torch.clamp(a1 @ self.W2.T + self.b2, min=0)
        return a2 @ self.W3.T + self.b3

st.title("ECG Heartbeat Classifier (PyTorch from Scratch)")
model = ECGNet()
model.load_state_dict(torch.load("results/best_model.pt", map_location="cpu"))
model.eval()

uploaded = st.file_uploader("Upload a CSV file with 1 row of 187 values", type=["csv"])
if uploaded is not None:
    df_up = pd.read_csv(uploaded, header=None)
    signal = df_up.iloc[0, :187].values.astype(np.float32)
    true_label = int(df_up.iloc[0, 187]) if df_up.shape[1] > 187 else None
else:
    test_df = pd.read_csv("data/mitbih_test.csv", header=None)
    row_idx = st.number_input("Pick a test-set row number", min_value=0, max_value=len(test_df)-1, value=0, step=1)
    signal = test_df.iloc[row_idx, :187].values.astype(np.float32)
    true_label = int(test_df.iloc[row_idx, 187])

x_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    logits = model(x_tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
    pred = int(probs.argmax())

st.subheader(f"Predicted Class: {CLASS_NAMES[pred]} ({probs[pred]*100:.2f}%)")
if true_label is not None:
    st.write(f"True Label: **{CLASS_NAMES[true_label]}**")

col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot([1000 * i / FS for i in range(187)], signal)
    ax1.set_xlabel("Time (ms)"); ax1.set_ylabel("Normalised Amplitude")
    ax1.set_title("Heartbeat Waveform")
    st.pyplot(fig1)
with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    ax2.bar(["N", "S", "V", "F", "Q"], probs * 100)
    ax2.set_ylabel("Probability (%)"); ax2.set_ylim(0, 100)
    ax2.set_title("Class Probabilities")
    st.pyplot(fig2)
