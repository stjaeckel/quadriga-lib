## Modeling Strategy by Material Class

### Class 1: Rigid Reflectors (concrete, glass, steel, etc.)

**Geometry:** Model as actual wall thickness. Since ε' << 1 (nearly perfect reflection), almost no energy enters the material anyway. The internal path is irrelevant.

**Example:** 200mm concrete wall → Box with 200mm thickness. The tiny amount of energy that enters will bounce around inside but won't escape back into the room.

### Class 2: Porous Absorbers (foam, fiberglass, curtains, carpet)

This is where it gets interesting. The P.2040 parameters we derived assume a semi-infinite medium with the Delany-Bazley attenuation. The RT tool will naturally handle the multiple internal bounces.

**Key insight:** The ε''(f) accounts for volumetric absorption. The thicker the object, the more passes through lossy material, the more absorption.

**Geometry recommendations:**

| Material | Typical Thickness | Geometry |
|----------|-------------------|----------|
| Thin carpet | 8-10mm | Flat box on floor |
| Thick carpet | 15-25mm | Flat box on floor |
| Acoustic foam panel | 50-100mm | Box mounted on wall (with air gap if applicable) |
| Fiberglass panel | 50-100mm | Box |
| Light curtain | 5-10mm | Thin box or curved surface |
| Heavy curtain (draped) | 20-50mm effective | Wavy/folded surface with ~3× fabric thickness |

**For the air-gap case (foam → air → concrete):** Model as three separate watertight bodies with appropriate ε for each.

### Class 3: Empirical Absorbers (furniture, people)

These are volumetric absorbers where the "effective thickness" d_eff in the model represents how deep sound penetrates before being absorbed.

**Critical:** The ε'' values assume energy is absorbed over approximately d_eff distance. If you model a person as a 10cm thick cylinder but d_eff = 30cm, you'll get less absorption than expected.

**Recommended geometries:**

| Object | Geometry | Dimensions |
|--------|----------|------------|
| Person standing | Cylinder or capsule | ~30cm diameter, 170cm height |
| Person seated | Box or ellipsoid | ~50×50×100cm |
| Audience (dense) | Rectangular volume | Per-person: ~50×50×120cm |
| Couch | Box | Actual dimensions, ~25cm depth minimum |
| Mattress | Box | Actual dimensions |
| Upholstered chair | Box | ~60×60×80cm |

## The Thickness-Absorption Relationship

For a ray passing through a lossy dielectric:

```
Energy_out / Energy_in = exp(-2 × α × d)
```

Where α is the attenuation constant derived from ε'':

```
α ≈ (ω/c) × ε'' / (2√ε')  [for ε'' << ε']
```

So doubling the object thickness roughly doubles the dB absorption (in the exponential sense).

## Practical Example: Carpet

Let's trace a ray hitting carpet:

**Thin carpet (8mm):**
```
Air → Carpet (8mm) → Concrete floor
      ↓
      Ray enters carpet (ε' ≈ 0.5, ε'' ≈ 0.3 at 1kHz)
      Travels ~8mm, loses some energy
      Hits concrete (ε' ≈ 2.4e-9), nearly total reflection
      Travels back through carpet, loses more energy
      Exits to air
```

**Thick carpet (25mm):** Same path but 3× more material traversal → significantly more absorption.

## Recommended Modeling Approach

```
┌─────────────────────────────────────────┐
│             ROOM VOLUME (air, ε=1)      │
│                                         │
│    ┌─────────┐     ╭──────╮             │
│    │ Couch   │     │Person│             │
│    │ (box)   │     │(cyl) │             │
│    │ 25cm    │     │ 30cm │             │
│    └─────────┘     ╰──────╯             │
│  ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄   Carpet   │
│  ████████████████████████████  (15mm)   │
├─────────────────────────────────────────┤
│          CONCRETE FLOOR (200mm)         │
├─────────────────────────────────────────┤
│                GROUND                   │
└─────────────────────────────────────────┘
```

Each object is a watertight body with:
- Outward-facing normals
- Appropriate ε'(f), ε''(f) from the P.2040 parameters
- Physical thickness that matches real-world dimensions

## Optimization Use Case

Assuming Monte-Carlo phase treatment:

1. **Walls/floor/ceiling:** Use actual architectural dimensions
2. **Acoustic treatment:** Use actual panel dimensions
3. **Furniture:** Use bounding-box approximations at real scale
4. **People/audience:** Use simplified cylinders/boxes at d_eff scale (~30cm diameter for standing, box for seated)

The energy transport will be correct, which is what matters for standing wave analysis and coverage optimization.