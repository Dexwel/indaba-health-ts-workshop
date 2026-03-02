# 🎯 Why Time Series in Health?

> *"A single blood pressure reading tells you where a patient is. A week of readings tells you where they're going."*

---

## The Problem with Static Snapshots

Most health AI systems treat each clinical encounter as an independent event. A model sees: age, lab values, diagnosis codes → predicts outcome. This ignores the most important dimension in medicine: **time**.

Consider what gets lost when you flatten a patient's record into a single row:

- Was the temperature rising or falling over the last 6 hours?
- Has the heart rate been trending up for 3 days before the crash?
- Did the patient's glucose stabilize after the insulin change — or keep climbing?

The *trajectory* is the signal. The snapshot is just noise.

---

## Three African Health Problems That Are Fundamentally Temporal

### 🦟 Malaria Outbreak Cycles

Malaria transmission is driven by rainfall, temperature, and mosquito density — all of which follow seasonal cycles. Historical incidence data, when treated as a time series, reveals predictable epidemic windows **4–6 weeks before** they peak.

A well-calibrated time series model can give district health officers enough lead time to pre-position bed nets, antimalarials, and health workers. A static model cannot.

### 🏥 ICU Patient Deterioration

In under-resourced ICUs — common across sub-Saharan Africa — nurses monitor multiple patients simultaneously. An early warning system that reads hourly vital signs (heart rate, SpO₂, respiratory rate, temperature, blood pressure) as a time series can flag deteriorating patients **hours before** clinical collapse.

This is the problem we will work on in this workshop.

### 💊 HIV/TB Treatment Adherence

A patient who took 90% of doses in Month 1 but only 40% in Month 3 has a very different prognosis from one with steady 70% adherence. Adherence *trajectories* predict viral rebound and resistance development. A time series model on pharmacy refill and clinic visit data can identify patients drifting toward non-adherence before they fail treatment.

---

## The Delayed Label Problem

Health time series have a subtle but critical challenge: **the label is often delayed**.

You get infected with sepsis-causing bacteria on Monday. Your vitals start subtly changing on Wednesday. The clinical diagnosis doesn't happen until Friday. If you train a model to predict "sepsis in the next 12 hours" using Wednesday's data labeled with Friday's diagnosis, you're training on the right signal.

But if you use the wrong alignment — labeling the entire record with the final outcome — you create **label leakage** that makes models look great on training data and fail completely in deployment.

> **This is the #1 reason health AI models fail when deployed. We will handle it correctly in the notebook.**

---

## Why African Researchers Have an Advantage Here

African health researchers have something rare: **genuine contextual expertise** about the specific failure modes of health systems on the continent.

- You know which data fields are systematically missing (and why)
- You know that a "normal" vital sign range calibrated on European populations may not hold
- You understand the resource constraints that determine whether a model is actually deployable
- You have access to African health datasets that global researchers don't

This isn't a disadvantage to overcome. It's a research edge to exploit.

---

*Next: [Core Concepts →](02-core-concepts.md)*
