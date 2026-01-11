# Perturbation-Robustness-Testing
These codes assume a simple scenario: a single-joint pendulum, for demonstration purposes. You can extend it to more complex robot models (such as robotic arms imported via URDF) as needed. The codes use Isaac Sim's core APIs, including omni.isaac.core and omni.isaac.kit.

This code will randomize multiple dimensional parameters (such as gravity, joint damping, friction, and mass perturbation), run multiple simulations, and check robustness indicators (for example, whether the fitting R² of joint torque is >= 0.9, which indicates the stability of the control strategy under perturbations). The success rate must be >= 80%.

**Instructions**:

• Multi-dimensional perturbations: including gravity, damping, friction, and mass. These can be adjusted via range parameters.

• Operation: Load a USD stage or URDF in Isaac Sim to replace the simple prim.

• Metrics: Use R² to check torque stability; it can be replaced with your control metrics (such as trajectory tracking error).

• Notes: scikit-learn needs to be installed. If the scene is complex, increasing the number of steps will prolong the running time.
