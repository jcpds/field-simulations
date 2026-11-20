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

- Toolbar: select/move, add metal ellipse, insulator ellipse, superconductor ellipse, in-plane coil, out-of-plane coil.
- Click blank canvas to add chosen shape. Drag objects to reposition.
- Select an object and drag the yellow circle to rotate it or the cyan squares to resize its width and height independently. The sidebar also lets you enter a rotation angle, edit size and metal charge, or adjust coil strength and polarity.
- Mouse wheel zooms about pointer. Middle-drag pans.
- Save/load scenes as JSON.
- Green shows electric field magnitude, red shows magnetic field magnitude, and yellow shows overlap. Use the display checkboxes and separate gains to inspect either field. In-plane coils label their north and south ends.

## Physical limitations (important)

This prototype shows qualitative, static electric and magnetic fields in a 2D slice. Fields and display gains use relative units.

The electric model redistributes sampled surface charge so that a metal is approximately equipotential. Each isolated metal or group of touching metals retains its combined assigned charge. A neutral metal develops opposite surface charges near a charged object. Insulators use an approximate bound-charge response with relative permittivity εᵣ = 10 and zero net charge.

In-plane coils use a pair of softened pole fields. Opposite poles facing across a gap reinforce the field there. Like poles oppose each other at the midpoint. The field near the outer ends also changes with spacing, orientation, and polarity. Out-of-plane coils use a simplified axial field model.

Superconductors model the ideal Meissner state. The magnetic field is zero inside the ellipse. Outside, equivalent internal magnetic sources approximate the screening currents that make the field tangent to the surface. The solve treats each ellipse as a slice through an ellipsoid whose depth radius is the smaller of its two drawn radii. Both coil types induce screening. In the electric model, a superconductor behaves as a conductor. This approximation excludes flux pinning, penetration depth, and critical fields.

These static approximations cover electrostatic induction and dielectric polarization. They omit time-dependent induction, eddy currents, forces, and torques.

**Safety:** A simulated field does not validate high-voltage hardware. Avoid high voltage and unfused high-current batteries in initial physical tests.
