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
- Select an object and drag its yellow handle to rotate it. The sidebar also lets you enter a rotation angle, edit size and metal charge, or adjust coil strength and polarity.
- Mouse wheel zooms about pointer. Middle-drag pans.
- Save/load scenes as JSON.
- Green shows electric field magnitude, red shows magnetic field magnitude, and yellow shows overlap. Use the display checkboxes and separate gains to inspect either field. In-plane coils label their north and south ends.

## Physical limitations (important)

This prototype shows qualitative, static electric and magnetic fields in a 2D slice. Fields and display gains use relative units.

The electric model redistributes sampled surface charge so that a metal is approximately equipotential. Each isolated metal or group of touching metals retains its combined assigned charge. A neutral metal develops opposite surface charges near a charged object. Insulators use an approximate bound-charge response with relative permittivity εᵣ = 2 and zero net charge.

In-plane coils use a pair of softened pole fields. Opposite poles facing across a gap reinforce the field there. Like poles oppose each other at the midpoint. The field near the outer ends also changes with spacing, orientation, and polarity. Out-of-plane coils use a simplified axial field model. These static approximations cover electrostatic induction and dielectric polarization. They omit time-dependent induction, eddy currents, forces, and torques.

**Safety:** A simulated field does not validate high-voltage hardware. Avoid high voltage and unfused high-current batteries in initial physical tests.
