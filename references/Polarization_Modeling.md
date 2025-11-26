# Polarization Modeling for MIMO Paths (Jones Formalism)

## Overview

Each propagation path is modeled by a 2×2 Jones matrix acting on the field vector in the linear basis $\{\hat H,\hat V\}$. This deterministic per-path model captures polarization mixing via complex gains and phases. A single common path phase represents the path delay and applies equally to both polarizations.

A single paramete, the cross-polarization ratio in the linear basis, $X=\mathrm{XPR}_{\text{lin}}$ (linear scale; e.g., $X=2$ for 3 dB), governs both:

1. linear↔linear crosstalk (H to V), and
2. circular↔circular crosstalk (LHCP to RHCP).

The construction also yields a predictable ellipticity when exciting the channel with a purely linear input.

## Implemented Model (per NLOS sub-path)

### Parameter mapping (magnitudes)

Given $X=\mathrm{XPR}_{\text{lin}}$ in linear scale, define magnitudes
$$
\alpha=\sqrt{\frac{X}{1+X}},\qquad
\beta=\sqrt{\frac{1}{1+X}},
$$
so that per column the co:cross power ratio is $\alpha^2:\beta^2 = X:1$ and $\alpha^2+\beta^2=1$.

### Phases

* **Common path phase:** draw $\phi_0 \sim \mathcal U[0,2\pi)$ and multiply the whole matrix by $e^{j\phi_0}$.
* **Within-column relative phase (linear ot circular control):** enforce quadrature $\Delta=\pi/2$ between co and cross terms in each column.
* **Absolute column phases:** use the same absolute phase for both columns, $\theta_d=\theta_a\equiv\theta$.

### Jones matrix in the linear basis

Up to the global phase (e^{j(\phi_0+\theta)}), the implemented matrix is
$$
J_{\text{lin}} = 
\begin{bmatrix}
\alpha  & j\beta \\
-j\beta & \alpha
\end{bmatrix}.
$$
This specific phase structure ensures (i) the requested linear XPR and (ii) the same XPR in the circular basis.



## Reasoning behind the construction

1. **Separation of propagation and polarization effects.**
   The common phase $e^{j\phi_0}$ captures delay; all polarization mixing is encoded by relative phases inside $J$.

2. **Direct control of XPR.**
   Choosing $\alpha,\beta$ from $X$ fixes the linear-basis co/cross power ratio exactly, independent of phases.

3. **Realistic ellipticity from linear inputs.**
   Quadrature within each column ($\Delta=\pi/2$) guarantees a well-defined circular component from a linear input, producing an ellipse with axial ratio tied to $X$.

4. **Basis consistency.**
   The specific sign pattern of the off-diagonal terms ($+j\beta$ and $-j\beta$) makes the circular-basis co/cross power ratio equal to the same $X$ used in the linear basis, yielding symmetric LHCP-RHCP leakage.

---

## Crosstalk Characteristics

### A) Linear - Linear (H↔V)

For an $H$ (or $V$) input, the output power splits as
$$
P_{\text{co}}=\alpha^2=\frac{X}{1+X},\qquad
P_{\text{cross}}=\beta^2=\frac{1}{1+X},
$$
so the linear-basis XPR is
$$
\mathrm{XPR}_{\text{lin}}=\frac{P_{\text{co}}}{P_{\text{cross}}}
= \frac{\alpha^2}{\beta^2}=X.
$$

### B) Linear - Circular (ellipticity from a linear input)

For a pure H/V input, the output Jones vector (up to a global phase) is $[\alpha , -j\beta]^T$.
Its Stokes parameters yield:

* **Circular fraction:** $S_3/S_0 = 2\alpha\beta$.
* **Ellipticity angle:** $\chi=\tfrac12\arcsin(2\alpha\beta)$.
* **Axial ratio (major/minor):**
  $$
  \mathrm{AR}=\frac{1}{|\tan\chi|}=\sqrt{X}.
  $$
  Thus with $X=2$ (3 dB XPR), a linear input emerges as an ellipse with $\mathrm{AR}=\sqrt{2}\approx1.41$.

### C) Circular - Circular (LHCP↔RHCP)

Transforming to the circular basis with the standard unitary $U$ gives (up to a global phase)
$$
J_{\text{circ}} = U J_{\text{lin}} U^{H} = 
\begin{bmatrix}
\alpha & -\beta \\
-\beta & \alpha
\end{bmatrix}.
$$
For an LHCP input $[1,0]^T$, the output has co-polar amplitude $\alpha$ and cross-polar amplitude $\beta$. Hence
$$
\mathrm{XPR}_{\text{circ}}
=\frac{\alpha^2}{\beta^2}=X,
$$
and the same holds for RHCP. **Conclusion:** the implemented model yields equal XPR in linear and circular bases, i.e., LHCP↔RHCP leakage at the specified ratio $X$.


## Practical Notes

* **Normalization & gains:** The expressions above use column normalization $(\alpha^2+\beta^2=1)$. In practice, multiply $J$ by the scalar **path gain** (including pathloss and delay-phase).
* **Statistical realism:** Across NLOS subpaths, randomize the common phase $\phi_0$, the shared absolute phase $\theta$, and (optionally) allow small fluctuations of $X$ per subpath around the cluster’s nominal XPR.
* **Depolarization vs. Jones:** A single Jones matrix is non-depolarizing. Cluster averages over randomized subpaths reproduce partial depolarization at the link level without leaving the Jones framework.


### Summary

The implemented NLOS Jones matrix enforces a single XPR parameter $X$ that simultaneously:

* fixes **linear** co:cross power at $X$,
* produces predictable **ellipticity** from linear inputs with $\mathrm{AR}=\sqrt{X}$,
* and yields **circular** co:cross power equal to $X$ (balanced **LHCP↔RHCP** leakage).

