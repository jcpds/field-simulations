# E/B Field Sandbox

Interactive Windows desktop prototype for exploring qualitative 2D electrostatic and magnetic field layouts.

## Setup

Python 3.10+ recommended. In VS Code terminal:

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Numba is optional; it accelerates field calculations using multicore CPU parallelism after the first JIT compilation. If installation fails, install only `PySide6 numpy` and the program uses NumPy.

## Controls

- Toolbar: select/move, add metal ellipse, insulator ellipse, in-plane coil, out-of-plane coil.
- Click blank canvas to add chosen shape. Drag objects to reposition.
- Click object to edit rotation, size, charge (metal) or coil strength/polarity.
- Mouse wheel zooms about pointer. Middle-drag pans.
- Save/load scenes as JSON.
- Green = magnitude of E, red = magnitude of B, yellow = overlap.

## Physical limitations (important)

This is a **qualitative, static, 2D approximation**, not a Maxwell solver. Charge is represented by sampled perimeter point charges, with an approximate curvature weighting; overlapping/touching metals are grouped and their assigned charge signs are summed. It does **not** rigorously enforce equipotential boundaries or surface-charge distribution, solve dielectric polarization, accurately model finite solenoid windings, include induction/eddy currents, or predict force/torque. A positive and negative charge placed on touching conductors can cancel in the approximation. B-field intensities are illustrative and are not calibrated to tesla. Colors have independent auto-scaled E/B display gains and cannot be used to infer field energy. Magnetic sources use dipole-like in-plane fields and a simplified out-of-plane field model. No claim of quantitative accuracy.

**Safety:** A simulated field does not validate high-voltage hardware. Avoid high voltage and unfused high-current batteries in initial physical tests.
