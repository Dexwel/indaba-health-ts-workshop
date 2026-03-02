# 📚 Core Concepts: Health Time Series

---

## What Makes Health Time Series Special

Health time series are not like stock prices or weather data. They have five properties that shape every modeling decision you'll make:

**1. Irregular Sampling**
Vitals are recorded when a nurse has time. Lab results arrive in bursts. Some patients have readings every 5 minutes; others have gaps of 8 hours. Most off-the-shelf time series models assume uniform intervals — this assumption fails immediately on real clinical data.

**2. Multivariate with High Missingness**
A patient might have 50 possible features. At any given time, 70% of them are missing. Not randomly missing — the *pattern* of missingness is itself a clinical signal. If a creatinine hasn't been ordered in 3 days, that's information.

**3. Non-Stationarity**
A patient's "normal" baseline shifts over their stay. A heart rate of 95 bpm might be worrying on Day 1 and reassuring on Day 5 (post-surgery). Models need to reason about *change from personal baseline*, not just absolute values.

**4. Extreme Class Imbalance**
Only 5–15% of ICU patients deteriorate to the outcome of interest. Your model will see 10+ negative examples for every positive. Training without handling this produces models that achieve 92% accuracy by predicting "fine" for everyone.

**5. Delayed and Noisy Labels**
As discussed in the motivation section, labels are often clinically defined hours or days after the underlying physiological event begins.

---

## The Method Landscape

### Classical: ARIMA

**When to use:** Univariate epidemiological forecasting (weekly malaria cases, monthly admissions). When you have limited data and need interpretability.

ARIMA (AutoRegressive Integrated Moving Average) decomposes a time series into trend, seasonality, and residual components. It's the first tool you should try for population-level health forecasting.

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(weekly_cases, order=(2, 1, 2))  # p=AR lags, d=diffs, q=MA lags
result = model.fit()
forecast = result.forecast(steps=6)  # predict next 6 weeks
```

**Limitation:** Doesn't handle multivariate inputs well. Can't capture complex nonlinear patterns.

---

### Sliding Window + Classical ML

**When to use:** Multivariate patient data, medium-sized datasets (< 10,000 patients), when you need fast iteration or explainability.

Convert a time series into a tabular problem by extracting statistical features over a rolling window: mean, std, min, max, trend slope. Feed to XGBoost or Random Forest.

**Why it often wins:** Surprisingly competitive with deep learning on small medical datasets. Fully interpretable via feature importance. SHAP values show which vital sign trend drove the prediction.

---

### LSTM / GRU

**When to use:** When temporal order and long-range dependencies matter. Patient deterioration over 24–72 hours. When you have > 5,000 sequences.

LSTMs use gating mechanisms (forget gate, input gate, output gate) to selectively remember and forget information as they read a sequence left to right. GRUs are simpler and often competitive.

```python
import torch.nn as nn

class PatientLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1]).squeeze(-1)
```

---

### Temporal Convolutional Network (TCN)

**When to use:** Long sequences where you need fast training. Dilated convolutions allow exponentially growing receptive fields — a TCN with 8 layers of dilation [1,2,4,8,16,32,64,128] sees over 500 timesteps with reasonable computational cost.

**Advantage over LSTM:** Parallelizable (no sequential bottleneck). Often trains 3–5× faster.

---

### Temporal Fusion Transformer (TFT)

**When to use:** Multi-horizon forecasting with known future covariates (e.g., scheduled medication doses, planned procedures). When interpretability via attention is important.

TFT produces attention weights that show *which past timesteps* most influenced each prediction — directly useful for clinical explanation.

---

## Evaluation Metrics That Matter in Health

| Metric | Use When |
|--------|----------|
| AUROC | Overall discrimination ability. Standard for imbalanced problems. |
| AUPRC | When positive class is rare (< 10%). More informative than AUROC in this regime. |
| Sensitivity @ 90% Specificity | Clinical framing: "At a false positive rate we can tolerate, how many true events do we catch?" |
| Early Warning Time | How many hours before clinical deterioration does our model first alert? |
| Calibration (Brier Score) | Does a "70% probability" prediction actually occur 70% of the time? |

> ⚠️ **Never** report just accuracy on a class-imbalanced health dataset. A model that always predicts "healthy" will have 95% accuracy and zero clinical value.

---

*Next: [Hands-On Coding →](03-hands-on.md)*
