# Acoustic Modes for ray_mesh_interact.cpp

## Overview

Add two new interaction types for acoustic wave simulation:
- **Type 4: Acoustic Reflection** — specular reflection using scalar acoustic coefficient
- **Type 5: Acoustic Transmission** — straight-through propagation with absorption, no refraction

These modes enable room acoustics simulation using the existing EM ray tracer with specially mapped material parameters (ε derived from acoustic impedance Z, where ε = (Z_air/Z_material)², resulting in ε << 1 for rigid materials).

## Key Differences from EM Modes

| Aspect | EM Modes (0,1,2) | Acoustic Modes (4,5) |
|--------|------------------|----------------------|
| Polarization | Full Jones matrix (2×2 complex) | Scalar complex coefficient |
| Reflection formula | Average of TE and TM | TE only |
| Total internal reflection | Yes (when ε₁/ε₂ causes evanescent wave) | Never — disabled entirely |
| Transmission (type 5) | — | Mirrors EM type 1: straight-through, no angle change |
| Output xprmatN | 8 elements (Jones matrix) | 2 elements used (Re, Im), rest zero |

## Required Changes

### 1. Disable Total Internal Reflection

For acoustic modes, bypass the total reflection condition entirely. Sound waves hitting high-impedance surfaces (mapped to ε << 1) reflect normally — they don't undergo TIR. Set `total_reflection = false` unconditionally for types 4 and 5.

### 2. Reflection Coefficient (Type 4)

Use only the TE reflection coefficient formula. Do not compute or average with TM. The TE formula:

$$R_{TE} = \frac{\eta_1 \cos\theta_1 - \eta_2 \cos\theta_2}{\eta_1 \cos\theta_1 + \eta_2 \cos\theta_2}$$

With the acoustic impedance mapping (η ∝ Z), this produces the correct acoustic pressure reflection coefficient. Set `reflection_gain = |R_TE|²` (no 0.5 averaging factor).

### 3. Transmission Coefficient (Type 5)

Use energy conservation at the interface: `refraction_gain = 1 - reflection_gain`. The in-medium attenuation (already implemented) handles absorption during traversal. This mirrors how EM type 1 works.

For the dense-to-light case (ray exiting absorber back to air), apply the same bypass as EM type 1: set T = 1, no additional interface loss. Absorption already occurred during traversal.

### 4. Gain Calculation

Type 4 uses `gain * reflection_gain` (same as type 0).
Type 5 uses `gain * (1 - reflection_gain)` (same as type 1).

### 5. Skip Polarization Basis Transformation

For acoustic modes, the entire polarization basis transformation block is unnecessary. Instead of computing the Jones matrix from incoming/outgoing basis vectors and TE/TM coefficients, directly output:

- **Type 4:** `amplitude * R_TE` as complex scalar
- **Type 5:** `amplitude * T_acoustic` where T_acoustic has magnitude `sqrt(1 - |R_TE|²)` and phase from T_TE

Write to first two elements of xprmatN only. Set remaining six elements to zero.

### 6. In-Medium Attenuation

No changes required. The existing loss tangent formulation works correctly with acoustic material parameters where ε'' encodes absorption from Delany-Bazley or empirical models.

## Output Format for Acoustic Modes

`xprmatN` layout for types 4 and 5:
- Element 0: Real part of scalar coefficient
- Element 1: Imaginary part of scalar coefficient  
- Elements 2-7: Zero (unused)

The scalar coefficient includes both the interface reflection/transmission and the in-medium attenuation (via `amplitude = sqrt(gain)`).

## Limitations

- **No acoustic refraction:** Type 5 assumes straight-through propagation. For scenarios requiring proper refraction angles inside materials, a future type 6 would be needed.
- **Angular dependence approximation:** The cos_theta2 calculation uses EM Snell's law, which is inexact for acoustics. For rigid reflectors (R ≈ ±1) this is irrelevant. For porous absorbers at highly oblique angles, some error exists but is secondary to the dominant in-medium absorption.
