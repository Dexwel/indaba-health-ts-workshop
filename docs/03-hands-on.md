# 💻 Hands-On Coding

> **40 minutes · Three progressive tasks · Google Colab**

---

## Before You Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

Click the badge above to open the workshop notebook. **Pre-run outputs are cached** — if your runtime disconnects, you can still follow along.

---

## The Dataset

We use a simulated ICU dataset modeled on the structure of **PhysioNet's Sepsis Prediction Challenge** (2019). Each patient record contains:

- **6 vital sign channels:** Heart rate, SpO₂, respiratory rate, temperature, systolic BP, diastolic BP
- **72 hourly timesteps** (3 days of monitoring)
- **Binary label:** Does the patient deteriorate within the next 12 hours?
- **~15% positive rate** (sepsis/deterioration)

```
Dataset shape: (8000 patients, 72 timesteps, 6 features)
Label: binary (0 = stable, 1 = deterioration within 12h)
Missing data: ~23% of values (MCAR + MNAR pattern)
```

---

## The Three Tasks

### [Task 1: Explore the Data →](03a-task1.md)
*10 minutes*
Load the dataset, visualize patient trajectories, understand missingness patterns, compute rolling statistics. Train your intuition before building any model.

### [Task 2: Sliding Window Baseline →](03b-task2.md)
*15 minutes*
Build a featurization pipeline, handle missing data, train an XGBoost classifier. Evaluate with AUROC and AUPRC. This baseline is deliberately simple — and often surprisingly powerful.

### [Task 3: LSTM Model →](03c-task3.md)
*15 minutes*
Build a PyTorch LSTM that reads the full 72-hour sequence. Train with early stopping. Compare head-to-head against the baseline. Understand *why* the sequential model does better — and when it doesn't.

---

## Group Working Mode

Work in pairs or groups of 3. One person drives the notebook, others read ahead and catch errors. After each task, discuss:

1. What surprised you about the results?
2. What would you try differently if you had another 30 minutes?

---

*Start with [Task 1 →](03a-task1.md)*
