# Perturbation-Robustness-Testing

**Reference trajectory (without disturbance)**

• Multi-dimensional disturbances: gravity / damping / joint friction / mass

• Robustness indicators:

• RMS trajectory error

• No divergence (bounded + non-NaN)

• Success rate ≥ 80%.

**Code Description (Robustness)**

Perturbation dimensions: gravity / damping / joint friction / mass

Reference trajectory: passive pendulum without perturbation

**Indicators**:

RMS(θ − θ_ref)

boundedness (prevention of numerical explosion)

**Significance**:

👉 Whether the control / dynamic model maintains predictability and stability under parameter uncertainty
