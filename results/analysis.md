# Assignment 03 - Results & Analysis

## 4.1 Table 1 - Data
| Metric | Value |
| :--- | :--- |
| Train / validation / test rows | 78,799 / 8,755 / 21,892 |
| Minimum / maximum signal value | 0.0000 / 1.0000 |
| Rows per class, train file (N, S, V, F, Q) | 72,471 / 2,223 / 5,788 / 641 / 6,431 |
| Rows per class, test file (N, S, V, F, Q) | 18,118 / 556 / 1,448 / 162 / 1,608 |

## 4.2 Table 2 - Network and Training
| Metric | Value |
| :--- | :--- |
| Trainable parameters | 32,645 (187x128+128 + 128x64+64 + 64x5+5) |
| Initial loss (before training) | 1.7413 |
| Gradient check: max difference for W^[3] / b^[3] | 2.98e-08 / 5.96e-08 |
| Chosen alpha | 0.5 |
| Best epoch (lowest validation loss) | 23 |
| Training / validation loss at best epoch | 0.0560 / 0.0722 |
| Training / validation accuracy at best epoch | 98.26% / 97.93% |
| Training time and hardware | 112.4s (cpu) |

## 4.3 Table 3 - Learning-Rate Choice
| alpha | Lowest validation loss | Validation accuracy at that epoch |
| :--- | :--- | :--- |
| 0.01 | 0.1618 (Epoch 30) | 95.79% |
| 0.1 | 0.0862 (Epoch 30) | 97.64% |
| 0.5 | 0.0722 (Epoch 23) | 97.93% |

## 4.4 Table 4 - Test Results
| Metric | N | S | V | F | Q |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Per-class accuracy | 99.60% | 66.37% | 91.37% | 62.96% | 97.95% |

| Summary Metric | Value |
| :--- | :--- |
| Test accuracy | 97.82% |
| Balanced accuracy | 83.65% |
| Always-'N' accuracy (Task 6.1) | 82.76% |

### Test Confusion Matrix (Rows = True, Columns = Predicted)
       N    S     V    F     Q
N  18045   21    18   11    23
S    181  369     4    0     2
V    100    4  1323   15     6
F     39    0    21  102     0
Q     27    1     5    0  1575

## Task 6 - Analysis Answers
- **6.1 Always-N Baseline:** Always predicting N gives 18,118 / 21,892 = 82.76% test accuracy, whereas our model reaches 97.82% test accuracy and 83.65% balanced accuracy (vs. 20.00% balanced accuracy for Always-N). Overall accuracy alone hides failure on minority arrhythmia classes.
- **6.2 Lowest Per-Class Accuracy:** Class F has the lowest accuracy (62.96%) and is most often mistaken for class N (39 times) because of overlapping waveform morphology and severe class imbalance.
- **6.3 Loss Curves:** Training loss decreases steadily while validation loss reaches its lowest value (0.0722) at epoch 23 with a small generalization gap, showing mild overfitting in late epochs prevented by saving the best validation checkpoint.
- **6.4 Zero Init vs He Rule:** Zero initialization makes all hidden neurons symmetric with identical gradients; He initialization scales by sqrt(2 / n^[l-1]) because ReLU zeros out half the pre-activations, preserving signal variance across layers.
- **6.5 Momentum & Weight Decay:** Momentum cancels opposite gradient components across consecutive mini-batches to dampen oscillations; weight decay shrinks weights with zero data gradient toward zero by factor (1 - alpha*(1-beta)*lambda).
- **6.6 Helping Rare Class F:** Use class-weighted cross-entropy loss (inverse class frequency weights) or oversample rare classes during mini-batch construction.
