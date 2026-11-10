# Classifying Heartbeats from ECG Signals — Neural Network from Scratch in PyTorch

An end-to-end multi-class arrhythmia classifier built and trained **from scratch in PyTorch** without ready-made layers (`nn.Linear`, `nn.ReLU`, `nn.Sequential`), built-in loss functions (`nn.CrossEntropyLoss`), or optimizers (`torch.optim`). Developed for **Machine Learning (Assignment 03, Fall 2026 — CLO4)** on single-heartbeat waveforms from the [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) via the [Kaggle ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).

## Project Motivation & Core Question

Electrocardiogram (ECG) bedside monitors must rapidly distinguish normal sinus beats from ectopic and fusion arrhythmias. However, clinical ECG repositories are heavily skewed toward normal heartbeats (`82.77%` of all recordings in MIT-BIH). This repository investigates a practical engineering question:

> *How well does a compact, hand-built neural network recognize each of the five heartbeat classes, and why is overall accuracy alone a dangerous metric to trust under severe class imbalance?*

### What Is Implemented from First Principles
- **Raw Parameter Matrices & He Initialization:** Every weight matrix $W^{[\ell]}$ and bias vector $b^{[\ell]}$ is registered directly as a `torch.nn.Parameter` initialized with $W^{[\ell]}_{ij} \sim \mathcal{N}\!\left(0, \sqrt{2 / n^{[\ell-1]}}\right)$ and $b^{[\ell]} = 0$ under `torch.manual_seed(0)`.
- **Custom Forward Pass & Clamped ReLU:** Mini-batch matrix affine transformations $Z^{[\ell]} = A^{[\ell-1]} W^{[\ell]\top} + b^{[\ell]}$ with manual ReLU activation $g(z) = \max(z, 0)$ via `torch.clamp(z, min=0)`.
- **LogSumExp Cross-Entropy & Analytical Gradient Verification:** Numerically stable softmax cross-entropy $\mathcal{L} = -z_y + \log \sum_{j=1}^{5} e^{z_j}$ using `torch.logsumexp`, validated against hand-derived output-layer gradients $\partial J / \partial W^{[3]}$ and $\partial J / \partial b^{[3]}$ to within **$2.98 \times 10^{-8}$** and **$5.96 \times 10^{-8}$**.
- **Manual Mini-Batch SGD with Momentum & L2 Weight Decay:** Custom velocity buffers $v_P = \beta v_P + (1 - \beta)g$ ($\beta = 0.9$) and weight-only L2 regularization ($\lambda = 10^{-4}$) executed inside `torch.no_grad()`.
- **Interactive Clinical Front-End:** Includes both a Streamlit inference app (`app/app.py`) and a self-contained, browser-executable clinical HTML workbench (`app/ecg_classifier_presentation.html`) embedding the trained `best_model.pt` weights.

## Dataset & Split Summary (Table 1)

Single heartbeats are sampled at **125 Hz** across **187 time steps** (`1,496 ms` window, normalized to `[0.0000, 1.0000]`) in columns `0..186`, with integer labels `0..4` in column `187`. The training CSV (`87,554` rows) is partitioned deterministically using `np.random.default_rng(42)` into `78,799` training rows and `8,755` validation rows (`10%`), while `mitbih_test.csv` (`21,892` rows) is held out strictly for final evaluation.

| Metric | Verified Value |
| :--- | :--- |
| Train / validation / test rows | `78,799` / `8,755` / `21,892` |
| Minimum / maximum signal value | `0.0000` / `1.0000` |
| Rows per class, train file (`N, S, V, F, Q`) | `72,471` / `2,223` / `5,788` / `641` / `6,431` |
| Rows per class, test file (`N, S, V, F, Q`) | `18,118` / `556` / `1,448` / `162` / `1,608` |

## Network Architecture & Training Summary (Tables 2 & 3)

`ECGNet` maps each 187-sample heartbeat through two hidden ReLU layers (`128` and `64` units) to `5` output logits:

$$\text{Total Parameters} = \underbrace{(187 \times 128 + 128)}_{24,064} + \underbrace{(128 \times 64 + 64)}_{8,256} + \underbrace{(64 \times 5 + 5)}_{325} = 32,645$$

| Network & Training Metric | Value |
| :--- | :--- |
| Trainable parameters | `32,645` (`187×128+128` + `128×64+64` + `64×5+5`) |
| Initial loss (before training, `seed=0`) | `1.7413` (close to $\ln 5 \approx 1.6094$) |
| Gradient check: max difference for $W^{[3]}$ / $b^{[3]}$ | `2.98e-08` / `5.96e-08` (threshold $< 10^{-5}$) |
| Chosen learning rate $\alpha$ | `0.5` |
| Best epoch (lowest validation loss) | `Epoch 23` |
| Training / validation loss at best epoch | `0.0560` / `0.0722` |
| Training / validation accuracy at best epoch | `98.26%` / `97.93%` |
| Training time and hardware | `112.4s` (`CPU`) |

### Learning-Rate Sweep (`30` Epochs, $B = 128$, $\beta = 0.9$, $\lambda = 10^{-4}$)

| Learning Rate $\alpha$ | Lowest Validation Loss | Validation Accuracy at Best Epoch |
| :--- | :--- | :--- |
| `0.01` | `0.1618` (Epoch 30) | `95.79%` |
| `0.10` | `0.0862` (Epoch 30) | `97.64%` |
| **`0.50` (Selected)** | **`0.0722` (Epoch 23)** | **`97.93%`** |

## Held-Out Test Set Results (Table 4 & Confusion Matrix)

Evaluated once on the `21,892` test rows using the saved Epoch 23 checkpoint (`results/best_model.pt`):

| Metric | N (Normal) | S (Supravent.) | V (Ventricular) | F (Fusion) | Q (Paced) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Per-class accuracy** | `99.60%` | `66.37%` | `91.37%` | `62.96%` | `97.95%` |

| Overall Evaluation Metric | Value |
| :--- | :--- |
| **Test accuracy** | **`97.82%`** (`21,414 / 21,892`) |
| **Balanced accuracy (macro average recall)** | **`83.65%`** |
| **Always-"N" baseline accuracy (Task 6.1)** | **`82.76%`** (`20.00%` balanced accuracy) |

### $5 \times 5$ Test Confusion Matrix (Rows = True Class, Columns = Predicted Class)

| True \ Predicted | Pred N | Pred S | Pred V | Pred F | Pred Q | Row Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **True N** | **18,045** | 21 | 18 | 11 | 23 | 18,118 |
| **True S** | 181 | **369** | 4 | 0 | 2 | 556 |
| **True V** | 100 | 4 | **1,323** | 15 | 6 | 1,448 |
| **True F** | 39 | 0 | 21 | **102** | 0 | 162 |
| **True Q** | 27 | 1 | 5 | 0 | **1,575** | 1,608 |

## Repository Structure

```text
├── data/                        # Place mitbih_train.csv and mitbih_test.csv here (not committed)
├── src/
│   ├── load_data.py             # Raw CSV loader returning DataFrames
│   ├── plot_beats.py            # Plots 1 example beat per class (results/example_beats.png)
│   ├── prepare.py               # Task 1: Deterministic train/val/test split (seed 42) & float32/long tensors
│   ├── model.py                 # Task 2 & 3.1: ECGNet (nn.Parameter weights, He init, clamp ReLU) & cross_entropy
│   ├── grad_check.py            # Task 3.2: Analytical output-layer backprop verification (< 1e-7 error)
│   ├── train.py                 # Task 4: Custom SGD + momentum + L2 decay, LR sweep, curves, checkpointing
│   ├── metrics.py               # Task 5.2: Custom accuracy, confusion_matrix, and per_class_accuracy
│   └── test_model.py            # Task 5: Evaluates results/best_model.pt on the 21,892 test rows
├── results/
│   ├── analysis.md              # Complete tables (Tables 1-4) and Task 6 analysis answers
│   ├── best_model.pt            # Saved PyTorch state_dict from Epoch 23 (alpha = 0.5)
│   ├── example_beats.png        # Figure 1: Five heartbeat classes waveform comparison
│   ├── lr_comparison.png        # Figure 2: Validation loss across alpha in {0.01, 0.1, 0.5}
│   ├── training_curves.png      # Figure 3: Training & validation loss/accuracy for alpha = 0.5
│   └── confusion_matrix.png     # Figure 4: 5x5 test set confusion matrix heatmap
├── app/
│   ├── app.py                   # Task 7.1 Streamlit web interface
│   └── ecg_classifier_presentation.html  # Standalone browser workbench & viva presentation deck
└── README.md                    # Setup, execution instructions, and results overview
```

## How to Run Everything

### 1. Environment Setup
Install the allowed dependencies in a Python 3.9+ virtual environment:

```bash
pip install torch numpy pandas matplotlib streamlit
```

### 2. Download the Dataset
Download `mitbih_train.csv` and `mitbih_test.csv` from the [Kaggle ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) and place both files inside `data/`:

```bash
mkdir -p data
# Place mitbih_train.csv and mitbih_test.csv inside data/
python src/load_data.py
```

### 3. Reproduce All Tasks in Order

```bash
# Task 1.5: Plot one example heartbeat per class -> results/example_beats.png
python src/plot_beats.py

# Task 1.6 & 1.7: Verify deterministic train/val/test tensor shapes
python src/prepare.py

# Task 3.2: Run manual backpropagation gradient check against PyTorch autograd
python src/grad_check.py

# Task 4: Run learning-rate sweep {0.01, 0.1, 0.5}, generate curves, and save results/best_model.pt
python src/train.py

# Task 5: Evaluate saved checkpoint once on the 21,892 held-out test rows
python src/test_model.py
```

### 4. Launch the Front-End Interfaces (Task 7.1)

**Option A — Standalone Interactive HTML Presentation & Live Inference Workbench (Zero Server Required):**
Open `app/ecg_classifier_presentation.html` directly in any modern browser. It embeds the trained `best_model.pt` weights and executes the `ECGNet` forward pass in real time on test-set rows or uploaded 187-sample CSV files.

**Option B — Streamlit Web Application:**
```bash
streamlit run app/app.py
```
