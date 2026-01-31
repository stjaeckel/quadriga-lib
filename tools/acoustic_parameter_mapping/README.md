# Acoustic Material Modeling for Radio-Wave Ray Tracing Tools

## Overview

This tool provides a framework for simulating **acoustic wave propagation** using **radio-wave ray tracing (RT) tools** designed for ITU-R P.2040 material models. The key insight is that acoustic waves at kHz frequencies have similar wavelengths to radio waves at GHz frequencies, enabling the reuse of sophisticated EM propagation software for acoustic planning.

### Primary Use Case

**Multi-speaker placement optimization** in event spaces, studios, and architectural acoustics:
- Minimizing standing waves at low frequencies (20–200 Hz)
- Optimizing phase coherence at higher frequencies (1–10 kHz)
- Modeling absorption from acoustic treatment, furniture, and audience

---

## Physical Foundation

### Wavelength Equivalence

The fundamental principle exploits the wavelength match between acoustic and electromagnetic waves:

| Domain | Wave Speed | Example Frequency | Wavelength |
|--------|------------|-------------------|------------|
| Acoustic | 343 m/s | 1 kHz | 0.34 m |
| Radio | 3×10⁸ m/s | 875 MHz | 0.34 m |

**Frequency scaling factor:**
```
f_radio = f_acoustic × (c_light / c_sound)
        = f_acoustic × 874,636
        ≈ f_acoustic × 875,000
```

### Frequency Mapping Table

| Acoustic (Hz) | Radio (GHz) | Wavelength (m) | Use Case |
|---------------|-------------|----------------|----------|
| 30 | 0.026 | 11.4 | Sub-bass |
| 63 | 0.055 | 5.4 | Bass |
| 125 | 0.109 | 2.7 | Low-mid |
| 250 | 0.219 | 1.4 | Mid |
| 500 | 0.438 | 0.69 | Mid |
| 1000 | 0.875 | 0.34 | Mid-high |
| 2000 | 1.75 | 0.17 | High |
| 4000 | 3.50 | 0.086 | High |
| 8000 | 7.00 | 0.043 | Air/brilliance |

---

## ITU-R P.2040 Material Model

The ITU-R P.2040 recommendation models building materials using frequency-dependent complex permittivity:

```
ε(f) = ε'(f) - j·ε''(f)
```

With power-law parameterization:
```
ε'(f) = a × f^b       (real permittivity)
σ(f)  = c × f^d       (conductivity, S/m)
ε''(f) = 17.98 × σ / f   (imaginary permittivity)
```

Where `f` is frequency in GHz.

### Fresnel Reflection

At a planar interface between air (ε=1) and material (ε), the reflection coefficient at normal incidence is:

```
R = (1 - √ε) / (1 + √ε)
```

For acoustic waves, the equivalent relationship uses acoustic impedance:

```
R = (Z - Z_air) / (Z + Z_air)
```

Where `Z = ρc` (density × speed of sound) in rayls.

---

## Mapping Framework

### The Core Challenge

Acoustic and electromagnetic waves have different relationships between material properties and wave behavior:

| Property | EM Waves | Acoustic Waves |
|----------|----------|----------------|
| Wave speed | c = c₀/√ε | c = √(K/ρ) |
| Impedance | η = η₀/√ε | Z = ρc |
| Independent parameters | 1 (ε) | 2 (ρ, c) |

Since P.2040 assumes μ=1 (non-magnetic materials), we cannot independently match both wave speed and impedance. **For room acoustics applications, we prioritize correct reflection behavior.**

### Mapping Strategy: Impedance Matching

We derive ε such that the Fresnel reflection coefficient matches the acoustic reflection:

```
√ε = Z_air / Z_material
ε' = (Z_air / Z_material)²
```

For typical building materials where Z >> Z_air, this gives ε' << 1.

**Important:** This means acoustic materials appear as "optically rarer" media (lower refractive index than air) in the EM model. Your RT tool must handle ε < 1 correctly, including total internal reflection scenarios.

---

## Material Classes

### Class 1: Rigid Reflectors

**Materials:** Concrete, glass, steel, brick, gypsum board, wood panels

**Characteristics:**
- Very high acoustic impedance (Z >> Z_air)
- Near-perfect reflection (|R| > 0.999)
- Negligible absorption
- Frequency-independent behavior

**P.2040 Parameters:**
- ε' = (Z_air/Z)² ≈ 10⁻⁸ to 10⁻¹¹
- b = 0 (frequency-independent)
- c = d = 0 (no loss)

**Model Accuracy:** Exact match for reflection coefficient.

### Class 2: Porous Absorbers

**Materials:** Acoustic foam, fiberglass panels, mineral wool, curtains, carpet

**Characteristics:**
- Impedance closer to air
- Significant frequency-dependent absorption
- Wave penetration into material

**Physical Model:** Delany-Bazley empirical model based on flow resistivity σ_f (rayls/m):

```
X = ρ_air × f / σ_f

Z_c/Z_air = 1 + 0.0571×X^(-0.754) - j×0.087×X^(-0.732)
α_m = (ω/c₀) × 0.189 × X^(-0.595)   [Np/m]
```

**P.2040 Parameters:**
- ε'(f) derived from |R| matching
- ε''(f) derived from absorption behavior
- Both have non-zero frequency exponents (b, d ≠ 0)

**Model Accuracy:** RMS error in |R| ≈ 0.13 (good at higher frequencies, some deviation at low frequencies)

### Class 3: Empirical Absorbers

**Materials:** Furniture (couches, chairs, mattresses), people, audience

**Characteristics:**
- Complex geometry prevents physical modeling
- Uses measured absorption coefficients (Sabine α)
- Frequency-dependent absorption

**Conversion Method:**
1. Target |R| = √(1 - α) from measured absorption
2. Derive ε' to match reflection magnitude
3. Add ε'' proportional to absorption for internal losses

**P.2040 Parameters:**
- ε' typically 0.01–0.1 (frequency-dependent)
- σ increases with frequency (d ≈ 1.2–1.4)

**Model Accuracy:** RMS error in |R| ≈ 0.21 (acceptable for energy transport modeling)

---

## Geometry Modeling Guidelines

### RT Tool Requirements

This framework assumes your ray tracing tool:
1. Simulates complete refraction paths through volumetric objects
2. Handles multiple internal reflections/bounces
3. Requires watertight meshes with outward-facing normals
4. Correctly handles ε < 1 (including total internal reflection)

### Object Geometry Recommendations

#### Rigid Surfaces (Walls, Floors, Ceilings)

Model with **actual architectural dimensions**. Since ε' << 1, almost no energy penetrates—internal geometry is irrelevant.

```
Example: 200mm concrete wall → Box with 200mm thickness
```

#### Porous Absorbers

Model with **actual physical dimensions**. The volumetric ε'' handles absorption naturally—thicker objects absorb more.

| Material | Geometry | Typical Thickness |
|----------|----------|-------------------|
| Thin carpet | Flat box | 8–10 mm |
| Thick carpet | Flat box | 15–25 mm |
| Acoustic foam | Box (on wall) | 50–100 mm |
| Fiberglass panel | Box | 50–100 mm |
| Light curtain | Thin box | 5–10 mm |
| Heavy curtain | Wavy surface | 20–50 mm effective |

**Air-gap configurations:** Model as separate watertight bodies:
```
[Foam panel] → [Air gap] → [Concrete wall]
   50mm          100mm         200mm
```

#### Empirical Absorbers (Furniture, People)

Model with dimensions that match the **effective absorption depth** (d_eff).

| Object | Geometry | Recommended Dimensions |
|--------|----------|------------------------|
| Person (standing) | Cylinder/capsule | Ø30 cm × 170 cm |
| Person (seated) | Box/ellipsoid | 50 × 50 × 100 cm |
| Audience (per person) | Box | 50 × 50 × 120 cm |
| Couch | Box | Actual dims, ≥25 cm depth |
| Upholstered chair | Box | ~60 × 60 × 80 cm |
| Mattress | Box | Actual dimensions |
| Wood table | Box | Actual dimensions |

### Scene Assembly Example

```
┌─────────────────────────────────────────────────┐
│              ROOM VOLUME (ε = 1, air)           │
│                                                 │
│   ┌──────────┐                                  │
│   │  Foam    │ ← Acoustic treatment (50mm)      │
│   │  Panel   │                                  │
│   └──────────┘                                  │
│                                                 │
│   ┌─────────┐      ╭────────╮                   │
│   │  Couch  │      │ Person │                   │
│   │  (box)  │      │ (cyl)  │                   │
│   │  25cm   │      │  30cm  │                   │
│   └─────────┘      ╰────────╯                   │
│                                                 │
│ ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄  Carpet (15mm)  │
├─────────────────────────────────────────────────┤
│            CONCRETE FLOOR (200mm)               │
└─────────────────────────────────────────────────┘
```

---

## Energy Transport Model

### Absorption Through Thickness

For a ray passing through a lossy dielectric of thickness d:

```
Energy_out / Energy_in = exp(-2 × α × d)
```

Where the attenuation constant α derives from ε'':

```
α ≈ (ω/c) × ε'' / (2√ε')   [for ε'' << ε']
```

**Implication:** Doubling material thickness approximately doubles the dB absorption.

### Multi-Layer Stacks

For configurations like `air → foam → air gap → concrete`:

1. Ray enters foam: partial reflection at air-foam interface
2. Ray traverses foam: exponential attenuation
3. Ray enters air gap: partial reflection at foam-air interface
4. Ray hits concrete: near-total reflection
5. Ray returns through stack: additional attenuation and interface losses

Your RT tool handles this naturally if each layer is a separate watertight body.

---

## Limitations and Caveats

### What This Framework Does Well

- Reflection magnitude at material interfaces
- Frequency-dependent absorption in porous materials
- Energy transport through multi-layer constructions
- Standing wave patterns from room geometry
- Early reflection analysis for speaker placement
- 
### What This Framework Does NOT Capture

- **Exact phase relationships** — Use Monte-Carlo phase modeling for interference
- **Diffraction** — Acoustic diffraction differs from EM; treat separately
- **Scattering from rough surfaces** — Not modeled in basic P.2040
- **Resonant absorbers** — Helmholtz resonators, membrane absorbers need special treatment
- **Structural transmission** — Flanking paths through building structure
- **Low-frequency modal behavior** — Below ~50 Hz, ray tracing assumptions break down

### Model Validity Range

| Parameter | Valid Range | Notes |
|-----------|-------------|-------|
| Frequency | 30 Hz – 8 kHz | Below 30 Hz, wavelength > room dimensions |
| Room size | > 3λ | Ray tracing requires room >> wavelength |
| Material thickness | > λ/10 | Thin layers need special treatment |
| Absorption coefficient | 0 – 0.95 | α > 0.95 extrapolates poorly |

---

## Usage

### Running the Converter

```bash
python acoustic_to_p2040.py
```

**Outputs:**
- `acoustic_p2040_materials.json` — Full parameter set with metadata
- `acoustic_p2040_materials.csv` — Tabular format for import

### Adding Custom Materials

Edit `create_material_database()` in the script:

```python
# Rigid material (need ρ and c)
materials.append(AcousticMaterial(
    name="custom_concrete",
    material_type=MaterialType.RIGID,
    rho=2500,      # kg/m³
    c=4000         # m/s
))

# Porous material (need flow resistivity)
materials.append(AcousticMaterial(
    name="custom_foam",
    material_type=MaterialType.POROUS,
    flow_resistivity=15000,  # rayls/m
    thickness=0.05           # m (for reference)
))

# Empirical material (need absorption coefficients)
materials.append(AcousticMaterial(
    name="custom_furniture",
    material_type=MaterialType.EMPIRICAL,
    alpha_coeffs=[0.15, 0.25, 0.40, 0.55, 0.60, 0.55],  # at 125,250,500,1k,2k,4k Hz
    d_eff=0.20  # effective absorption depth (m)
))
```

### Using Parameters in Your RT Tool

For each material at radio frequency f_GHz:

```python
epsilon_real = a * (f_GHz ** b)
sigma = c * (f_GHz ** d)
epsilon_imag = 17.98 * sigma / f_GHz
epsilon_complex = complex(epsilon_real, -epsilon_imag)
```

---

## Validation

The script includes validation that compares:
- **Acoustic |R|** — From physical models (impedance or Delany-Bazley)
- **P.2040 |R|** — From derived ε parameters

### Typical Validation Results

| Material Class | Avg RMS Error in abs(R) | Notes |
|----------------|---------------------|-------|
| Rigid | 0.0000 | Exact match |
| Porous | ~0.13 | Best at high frequencies |
| Empirical | ~0.21 | Acceptable for energy modeling |

### Interpreting Errors

For **speaker placement optimization**, errors in |R| of 0.1–0.2 are acceptable because:
1. Real-world absorption coefficients have similar measurement uncertainty
2. Monte-Carlo phase treatment averages out coherent errors
3. Standing wave patterns depend more on room geometry than exact α values

---

## References

### Standards
- **ITU-R P.2040-3** (2023): Effects of building materials and structures on radiowave propagation above about 100 MHz

### Acoustic Models
- Delany, M.E. & Bazley, E.N. (1970). "Acoustical properties of fibrous absorbent materials." *Applied Acoustics*, 3(2), 105-116.
- Bies, D.A. & Hansen, C.H. (2009). *Engineering Noise Control: Theory and Practice*, 4th ed.

### Material Properties
- Acoustic impedance values from standard engineering references
- Absorption coefficients from manufacturer data and acoustic measurement databases

---

## License

Apache-2.0 License

---

## Version History

- **v1.0** — Initial release with rigid, porous, and empirical material support
