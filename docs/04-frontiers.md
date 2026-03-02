# 🔭 Frontiers & Open Problems

> The research questions in this section are tractable, publishable, and underexplored. They are yours to take.

---

## Frontier 1: Uncertainty Quantification with Conformal Prediction

In clinical practice, a model that says **"70% probability of deterioration"** is more useful than one that simply says **"yes/no"**. But not all uncertainty methods are equally trustworthy.

**Conformal Prediction** offers something unusual: *distribution-free coverage guarantees*. If you calibrate on a held-out set, conformal prediction guarantees that your prediction sets contain the true label at least (1−α)% of the time — regardless of the underlying data distribution.

```python
from sklearn.model_selection import train_test_split
import numpy as np

# After training your model, get calibration scores
cal_scores = 1 - model.predict_proba(X_cal)[:, y_cal]  # nonconformity scores

# Set desired coverage (90%)
alpha = 0.10
threshold = np.quantile(cal_scores, 1 - alpha)

# At test time: prediction set includes all labels where score <= threshold
def predict_set(x):
    scores = 1 - model.predict_proba(x.reshape(1,-1))[0]
    return [label for label, score in enumerate(scores) if score <= threshold]
```

**Clinical implication:** A model can say "I'm not sure — both 'stable' and 'deteriorating' are within my 90% prediction set." This triggers a human review rather than a high-confidence false negative.

**Open question:** How do you maintain coverage guarantees when the patient population shifts between hospitals, or over time as clinical protocols change?

---

## Frontier 2: Federated Learning for Health Time Series

African health data is distributed across hundreds of small, siloed hospital information systems. Each hospital has too few patients to train a good model alone. But data sharing across institutions faces legal, ethical, and infrastructure barriers.

**Federated Learning** trains a shared model without moving data. Each hospital trains locally on its own patients, shares only model weight updates (not data), and a central server aggregates them.

```
Hospital A (Nairobi, 200 patients) ──┐
Hospital B (Lagos, 150 patients)  ──┤──→ [FedAvg Aggregation] ──→ Global Model
Hospital C (Accra, 300 patients)  ──┘
```

**Why this matters for Africa:** Many hospitals use offline EHR systems with intermittent internet. Federated algorithms need to handle **stragglers** (slow or unavailable clients) and **heterogeneous data distributions** (different patient populations, different equipment, different missing data patterns).

**Open research questions:**
- How do you handle the extreme non-IID case where Disease X is common in Hospital A and rare in Hospital B?
- Can differential privacy guarantees be maintained at the communication bandwidth constraints common in rural African hospitals?
- How do you audit a federated model for fairness across sites?

---

## Frontier 3: Foundation Models for Health Time Series

In 2024, pre-trained foundation models for time series arrived. Google's **TimesFM** was trained on 100 billion time points from diverse sources. **Moirai** (Salesforce) and **Chronos** (Amazon) followed.

The promise: fine-tune on your 500-patient African dataset and get a model that already "knows" what temporal patterns look like, dramatically reducing data requirements.

**The catch:** These models were pre-trained on financial and weather data, not clinical vitals. The distribution mismatch is significant. A heart rate that jumps from 60 to 140 bpm is not the same kind of anomaly as a stock price doubling.

**Open questions:**
- How much clinical pre-training data is needed before general time series pre-training helps rather than hurts?
- Can multi-task pre-training across African health datasets (malaria surveillance + ICU vitals + HIV adherence) build a useful "African health time series" foundation model?
- What does transfer learning look like when the source domain (ICU in Cape Town) and target domain (rural clinic in Tanzania) have fundamentally different monitoring practices?

---

## The Three Questions We Leave You With

These are not rhetorical. They are open problems worth a research paper each:

**Q1 — Label Imbalance Under Distribution Shift**
How do you handle extreme label imbalance in rare disease prediction when the positive rate shifts between training and deployment settings — which happens constantly in African health systems?

**Q2 — Cross-Site Generalization**
How do you build models that are robust to the distribution shifts that occur when a model trained in Nairobi gets deployed in Lagos? The patients are different, the equipment is different, the nurses' documentation practices are different.

**Q3 — Clinical Trust and Explainability**
How do you build clinical trust in a model that reads 72 hours of vitals and tells a nurse someone will crash? What explanation format is actually useful to a nurse making a split-second decision?

---

## Group Discussion: 10 Minutes

Before we close, share with the group:

> *"Given what you've built today, what is one thing you would do differently — and one thing you would actually try to publish?"*

---

*Back to [Resources →](05-resources.md)*
