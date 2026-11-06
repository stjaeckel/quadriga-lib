# Tutorial series: random fields and the sum-of-sinusoids method

Project plan for the tutorial series on random fields and the sum-of-sinusoids method. It describes what the series covers, the steps to get there and the decisions taken on the way, and is updated as the work goes on. How tutorials are written, built and published in general is described in [tutorials/README.md](../tutorials/README.md); this plan holds only what is specific to this series.

## 0. Status

- Part 1 is drafted (`tutorials/sos_01_random_fields.ipynb`), reviewed, and its references are checked against the full texts. The research behind the series is in section 5, the decisions in sections 4 and 6.
- Part 2 is drafted (`tutorials/sos_02_custom_acf.ipynb`), reviewed, and its references are checked against the full texts; scope, decisions and findings are in section 7. Its statements on the residual beyond the fitted range and on 1-D fits describe the library as it is; they are to be revisited when https://github.com/stjaeckel/quadriga-lib/issues/22 and https://github.com/stjaeckel/quadriga-lib/issues/24 are resolved.
- Next: the setup section of the tutorial index (input in section 5.6).

## 1. Goal

A series of three Jupyter tutorials on random fields and the sum-of-sinusoids (SOS) method, published on the website as `tutorials/sos_<nn>_<topic>.ipynb`. They were the first tutorials; the general rules learned while writing them are in `tutorials/README.md`.

## 2. The three parts

### Part 1: Introduction to random fields and the SOS method (beginner)

Notebook stem `sos_01_random_fields`. Possible now; all functions exist. Python only.

- What random fields are and where they emerge (introduction and theory).
- Which methods exist to generate them.
- What an autocorrelation function (ACF) is.
- The SOS method: basic idea and formulas, advantages and disadvantages against other methods.
- Generating spatially correlated numbers with the quadriga-lib API (basic calls): Normal, Uniform and integer values (`sos.randn`, `sos.rand`, `sos.randi`).
- Selecting a preset.
- Repeatable simulations with phases or seeds.
- Plots: two example processes side by side (Gauss and Exp preset), like Fig. 1 of the publication; desired and approximated ACF, at least in 1-D, in 2-D if a few lines of plot code do it (like Fig. 2).

### Part 2: Custom correlation functions (advanced)

Notebook stem `sos_02_custom_acf`; title "Custom correlation functions and the limits of the sum-of-sinusoids method". The content as written is in section 7.

- Approximating a custom ACF with the quadriga-lib tools (`sos.generate`, `sos.calc_mse`, `sos.acf_approx`, `sos.acf_estimate`).
- Using custom frequencies instead of the presets. The model has no weights: all sinusoids have the amplitude sqrt(2/N), and the only free parameters are the frequency vectors. The low-level calls (`sos.val`, `sos.map`, `sos.remove_bias`) take custom frequencies already; `sos.randn`, `sos.rand`, `sos.randi` and the GBSM functions take only a preset name.
- An extreme example: fully correlated from 0 to 10 m and uncorrelated beyond. This step is not a valid ACF in any dimension, not even in 1-D: points at 0, 6 and 12 m would need equal values at 0 and 6 m and at 6 and 12 m, but uncorrelated values at 0 and 12 m (the correlation matrix has determinant −1). Its spectrum has negative parts (Bochner's theorem), so it cannot be met exactly. An SOS generator always has a valid ACF, so the fit gives a valid approximation. The tutorial says this up front and uses it to show the limits. The triangle (1 − d/a) for d < a is valid in 1-D but not in 2-D, but only mildly: its 2-D spectrum dips to about −2 % of its peak, and fits follow it (section 7). The damped cosine exp(−d/a)·cos(2πd/b) shows the dependence on the dimension clearly.
- How well it is approximated with different numbers of sinusoids, and how the resulting field looks. The fitting figures of the publication (Fig. 3 to 6) belong here.
- Limitations of the method.

### Part 3: Spatial consistency in wireless channels (advanced)

Notebook stem `sos_03_spatial_consistency`. Not now: needs the finished GBSM implementation.

- How 3GPP TR 38.901 defines it: LSF autocorrelation, SSF autocorrelation, full spatial consistency.
- Which parameters of the parameter tables govern it.
- How the SOS method is integrated into the GBSM API.
- How to adjust the settings (presets, values).
- Example: how different preset choices change the results with identical parameters; closely spaced UEs are expected to behave differently.
- Undecided: the ensemble bias. UEs dropped close together have correlated parameters, so the whole ensemble can drift away from zero mean; `sos.remove_bias` corrects it. This was an issue in many simulation runs.

## 3. Steps

1. Kick-off: plan and decisions (this file).
2. Literature and web research by agents, before any code. One agent per topic, one report each; the reports are aggregated into section 5.
   1. Other methods for generating random fields, and how other simulation tools do it (e.g. ocean waves, pressure fluctuations over time).
   2. Textbook answers, with references, to the questions of part 1.
   3. How other simulators implementing TR 38.901 generate spatially correlated LSPs, and whether they support spatial consistency: Sionna, ns-3 (incl. 5G-LENA), MATLAB 5G Toolbox, OpenAirInterface, Vienna 5G simulators, others found.
   4. Later papers citing the 2018 SOS paper: what the authors used it for, which problems they solved. Hints for what to emphasise or which further tutorials to write.
   5. Fields beyond radio where random fields matter and where this implementation could help; their problems. Includes dual mobility (both ends moving, 6-D correlation): use cases in wireless and elsewhere, to decide where it goes in the series.
   6. Good practice for the environment set-up of Python tutorials on Linux and Windows: venv with pip (with Ubuntu system packages as an option) and a conda path, as other projects document it.
3. Summary of the findings, recorded in section 5.
4. Scope and content of part 1.

## 4. Kick-off decisions

- Versions: part 1 needs v0.13, the next release.
- Language: Python first. A MATLAB version follows as a separate project.
- Audience: part 1 is generic; the audience of the library are wireless engineers, but the SOS method is useful outside wireless too.
- Theory: brief, enough for a university-level engineer to follow, no walls of formulas, standard references. The focus is the software and how it helps; the theory is the groundwork.
- Comparison with other methods: one simple example of a few lines, using common Python libraries the reader may already have.
- Dual mobility: placement open (section 5.5).
- Fits with 1024 sinusoids may be left out of the notebooks to keep the run time short; they are examples, and research runs have more time.
- Names: `sos_01_random_fields`, `sos_02_custom_acf`, `sos_03_spatial_consistency`; they may change.

## 5. Research findings

Six research topics, done in November 2026 (section 3, step 2). The references of the part 1 notebook were checked against the full texts on 2026-11-06; the corrections from that check are included below. Keys in square brackets point to the reference list in section 5.8. A reference marked "(abstract)" was seen only as an abstract or a database record; its content must be checked against the full text before a tutorial quotes it. Numbers marked "(measured)" come from small checks run for this research, with numpy 2.3 or a local build of quadriga-lib (main branch); each says what was computed, so it can be repeated. Statements marked "(derived)" are short derivations, not taken from a source.

### 5.1 Methods for generating random fields

- Two families. Matrix methods factor the covariance matrix of the points: Cholesky [RW06, App. A.2], [KB13, Sec. 2], Karhunen-Loève expansion [SN26], sequential Gaussian simulation as in gstat [gstat], and Gaussian Markov random fields with the SPDE approach [LRL11]. They are exact or nearly exact on a fixed set of n points, but need memory ∝ n² and time up to n³ for the factorisation, and new points need new work. (measured: numpy Cholesky of an exponential covariance matrix on 1000, 2000 and 4000 random points took 0.04, 0.13 and 0.74 s CPU time, with 8, 32 and 128 MB for the matrix.)
- Spectral methods add up waves. Rice's random-noise representation [Ric44, §2.8], the spectral representation method [SD91] (abstract), [SDH87], Kraichnan's randomization method [Kra70], used as the default generator of GSTools [GST22, Sec. 2.3], [GST-src], turning bands [MW82] (abstract), [EL06] (abstract), Pätzold-type SOS in channel modelling [PKL96], [Pae11], FFT filtering and circulant embedding [KB13, Sec. 2], [DN97], ocean-wave synthesis [Tes04, Sec. 4.3-4.4], and sparse-spectrum phase screens [Cha19]. (derived) They differ in three choices: where the frequencies sit (regular grid, random, fitted), whether the amplitudes are fixed or random, and whether the sum is evaluated by FFT on a grid or directly at each point.
- Other fields almost all use spectral methods:
  - Ocean waves: FFT with random complex amplitudes in graphics [Tes04, Eqs. 39-43]; inverse FFT with random phases in OpenFAST SeaState, whose wave field repeats after the period `WaveTMax` [OF-SS].
  - Wind: TurbSim factors the spatial coherence matrix by Cholesky per frequency and then transforms to time by FFT (`Coh2H`, `LAPACK_potrf` in `TSsubs.f90` [OF-TS]), after Veers [Vee88] (abstract); Mann turbulence boxes are generated on FFT grids [Man98] (abstract), [Hip].
  - Atmospheric optics: FFT phase screens with added subharmonics for the low frequencies (`ft_sh_phase_screen` in AOtools [AOT-src]), or sparse spectra [Cha19] (abstract).
  - Geostatistics: GSTools uses the randomization method with 1000 modes by default [GST-src]; gstat uses sequential simulation [gstat]; TBSIM uses turning bands [EL06] (abstract).
- quadriga-lib's SOS has fixed amplitudes sqrt(2/N), fitted frequencies and direct evaluation at each position [JRBT18, Sec. II-III], [QL-API]. Its closest relatives are GSTools' `RandMeth` (random frequencies, random amplitudes, direct evaluation) [GST-src] and sparse-spectrum phase screens [Cha19].
  - Strengths [JRBT18, Sec. V, Table I], [QL-API]: no grid and no domain limit; memory of (D+1)·N numbers per generator (D frequency components and one phase per sinusoid); repeatable from a seed; positions can be added at any time; a two-device mode.
  - Weaknesses: the ACF is approximate [JRBT18, Sec. V]; the steep slope of the exponential ACF at the origin is not reproduced [JRBT18, Sec. V "Exponential ACF approximation"]; the cost per position grows with N; the values are not Gaussian for small N (section 5.2).
- FFT grids are periodic. (measured: on a 512 × 512 FFT-filtered grid with 1 m spacing, points at opposite edges, 255.5 m apart, still have a correlation of 0.96.) This is the reason for double-size grids in [Man98] and for the repetition in [OF-SS].

### 5.2 Textbook answers for part 1

- A random field is a random function of a multi-dimensional position; a random process is the special case of a one-dimensional index such as time [KB13, Sec. 1], [AT07] (abstract), [Van10]. [RW06, Sec. 2.2, Def. 2.1] defines the Gaussian process, not the random field in general.
- The SOS idea is old: Rice represented random noise as a sum of cosines with fixed amplitudes and random phases and used the central limit theorem for Gaussianity [Ric44, §2.8]. The spectral representation method of Shinozuka and Deodatis followed in structural engineering [SD91], [SD96] (both abstract), Pätzold's deterministic SOS in channel modelling [PKL96], [Pae11]. A sum with constant amplitudes and frequencies and random phases is both stationary and ergodic [HGPC13] (abstract), [Pae04].
- Wiener-Khinchin: the ACF of a stationary field and its power spectral density are a Fourier pair [RW06, Sec. 4.2.1, eq. 4.6]. Bochner's theorem: a function is a valid (positive-definite) stationary ACF if and only if it is the Fourier transform of a non-negative spectrum [RW06, Thm. 4.1]. (derived) An SOS generator has a line spectrum with non-negative weights, so its ACF is always valid; fitting an invalid target gives a valid approximation.
- Invalid targets: the step "1 up to 10 m, 0 beyond" is invalid in every dimension. (measured: points at 0, 6 and 12 m give the correlation matrix [[1,1,0],[1,1,1],[0,1,1]], with determinant −1 and smallest eigenvalue −0.414.) The triangle (1 − d/a) for d < a is valid in 1-D [RW06, eq. 4.21] but not in 2-D (measured: a negative eigenvalue of its covariance matrix on a 2-D grid). Isotropic correlations cannot be lower than −0.403 in 2-D or −0.217 in 3-D [SH09, Sec. 2].
- "Correlation length" has at least four conventions: the distance where the ACF falls to 1/e, stated in [JRBT18, Sec. II] and used by quadriga-lib [QL-API], which is the correlation distance d_cor of the exponential ACF of 3GPP [3GPP-901-19, clause 7.4.4, eq. 7.4-5] (3GPP cites ITU-R P.1816 for it and does not state 1/e in words); the distance where the ACF falls to 1/2 [WTN05]; Gudmundson describes the exponential decay by the correlation at a reference distance instead, e.g. 0.82 at 100 m [Gud91, eqs. 2-3, Fig. 1]; the length parameter ℓ of a kernel [RW06, Sec. 4.2.1]; the integral scale [GST]; and the scale of fluctuation 2∫ρ [Van10], [WW07]. (measured) For the same ℓ, the 1/e distances of the common models differ by up to 41 %.
- Smoothness: the exponential ACF gives a continuous but not differentiable field, the Gaussian ACF an infinitely differentiable one; the Matérn family lies between, with a field that is k times mean-square differentiable if and only if ν > k [RW06, Sec. 4.2.1]. A covariance matrix with a Gaussian ACF is nearly singular, so a Cholesky comparison needs a small diagonal term [RW06, App. A.2]. (measured: Cholesky fails without it for 400 points along a 100 m line with a Gaussian ACF.)
- Ergodicity: the mean of one realization over a finite domain scatters around the ensemble mean, with a variance given by Vanmarcke's variance function [Van10], [WW07, eq. 4]. (measured: for an exponential ACF the standard deviation of the mean over a track of 10 decorrelation distances is 0.42 σ, over 100 decorrelation distances 0.14 σ.) This leads to the ensemble bias of part 3.
- Finite N: the values of a sum of N cosines with amplitude sqrt(2/N) are bounded by sqrt(2N) (derived), and the excess kurtosis is −1.5/N (derived from the kurtosis 1.5 of one cosine with random phase; measured −0.37 at N = 4, −0.09 at N = 16 and −0.0064 at N = 256). With frequencies drawn at random (plain Monte Carlo), the ACF of the generator does not decay to zero: its rms at large distances is about sqrt(1/(2N)), 0.044 at N = 256 (measured). The fitted presets of quadriga-lib are not yet checked for this.
- Further reading for the tutorial: [RW06, ch. 4] (free PDF); [Pae11] for wireless readers; [SD91] for readers from other fields.

### 5.3 Spatial consistency in other TR 38.901 simulators

- TR 38.901 generates spatially consistent variables as "a 2D random process (in the horizontal plane) given the UT locations" [3GPP-901, clause 7.6.3.1], with correlation distances per scenario [3GPP-901, Table 7.6.3.1-2], and defines two procedures for moving UEs [3GPP-901, clause 7.6.3.2] and spatially consistent LOS states [3GPP-901, clause 7.6.3.3].
- Four methods are in use for spatially correlated LSPs:
  - (a) A first-order update per link along the trajectory, ρ = exp(−Δd/d_cor) [36.885, Annex A.1.4]: ns-3 default shadowing (`GetShadowing` [ns3-pl]), Simu5G (`computeShadowing` [Simu5G]), WiLabV2Xsim [V2X], 5G-air-simulator [air].
  - (b) An exp(−d/D) covariance matrix over the UEs of one drop, factored by Cholesky or matrix square root: Sionna (`_compute_lsp_spatial_correlation_sqrt` [Sio-lsp], `spatial_consistency_matrix_sqrt` [Sio-sc]).
  - (c) White noise on a 2-D grid filtered with an exponential kernel, as in the WINNER code (`LScorrelation.m` [WIN]): the MATLAB `h38901Channel` example helper (5 m pixels, nearest pixel; `generateLSP`, `getGridResolution` [MW-h]), NYUSIM [NYU19, Sec. III], and ns-3.49's new `SpatialGaussianField` [ns3-sgf].
  - (d) An FFT filter evaluated at the user positions: the Vienna 5G system-level simulator [VieSLS, Sec. 4.1.2], [VieWSA] (abstract). (derived) This is a sum of sinusoids on a regular frequency grid.
- Clause 7.6.3 coverage:
  - MATLAB `h38901Channel` helper: static consistency, procedures A and B, soft LOS [MW-h].
  - ns-3.49: drop-based fields (opt-in) and procedure A since ns-3.47; no procedure B, no soft LOS [ns3-ch], [ns3-doc], [ns3-rel].
  - Sionna v2.1 and newer: static consistency (opt-in) for LOS state, O2I distance and cluster and ray variables; the documentation states that the mobility procedures of clause 7.6.3.2 are not implemented and that fields are not kept across topology updates [Sio-rel], [Sio-slc], [Sio-rays].
  - OpenAirInterface [OAI-doc], the Vienna link-level simulator [VieLLS, ch. 10] and MATLAB CDL/TDL [MW-cdl]: no LSP correlation.
- No tool surveyed correlates base-station sites with each other (all correlate UE-UE per site), and none has a joint field over two moving ends; where both ends move, only the relative displacement is used [ns3-pl], [36.885], [Sio-cc].
- ns-3.49 rejects SOS in its source: the header of `SpatialGaussianField` states that a sum-of-sinusoids field "(the QuaDRiGa approach)" carries a ripple of about 0.05 at every lag, and that its own field stays within about 0.04 of exp(−d/d_cor) [ns3-sgf]. The tutorials should answer this with measured ACF errors against N.
- Grid-filter pitfalls:
  - Filtering white noise with an exp(−r/d) kernel does not give an exp(−r/d) ACF. (measured: for the radial kernel printed in [NYU19], the exact kernel autocorrelation at d = 10 m is 0.94 / 0.81 / 0.50 at 5 / 10 / 20 m, against 0.61 / 0.37 / 0.14 for exp(−r/d).) ns-3 corrects the decay length (λ = d_cor/1.996 [ns3-sgf]); the code of the MATLAB helper shows no correction [MW-h].
  - Filtering axis by axis makes the field anisotropic. (measured for the WINNER code [WIN]: 0.369 at 10 m along an axis, 0.247 at 9.9 m along the diagonal, against 0.372 for an isotropic field.) The same is inferred, not measured, for ns-3.49.
  - Updates along a trajectory depend on the path: a UE that returns to its start does not get its start value, as the ns-3 documentation states [ns3-doc].
- Not checked: the internal MATLAB package behind the current `h38901Channel`; the code of the Vienna system-level simulator and NYUSIM (not public, findings from their papers); the values of Table 7.6.3.1-2 were read from the Sionna and ns-3 code and must be checked against the official TR.

### 5.4 Papers citing the 2018 SOS paper

- The paper is [JRBT18], DOI 10.1109/GLOCOMW.2018.8644265, arXiv 1808.04659. 35 citing works were found: OpenAlex lists 29 [OpenAlex], Semantic Scholar 30 [S2], OpenCitations 25 [OCit], Crossref gives a count of 29 [Crossref]. Google Scholar was not reachable, so theses and non-English works are probably missing.
- About a quarter deal with the method itself. Five extend, compare or replace it:
  - Pessoa et al. derive the SOS coefficients analytically for single and dual mobility in 3-D and report lower approximation errors (13 to 21 dB) [Pes20] (abstract), and propose a positive-semidefinite ACF with negative correlation for 3-D processes [Pes21] (abstract). They criticise the dual-mobility simplification of [JRBT18] (citation context in [S2]).
  - Huang et al. fit the coefficients with a perceptron and extend from 3-D to 6-D, calling 6-D with arbitrary ACFs an open problem [Hua23] (abstract).
  - Wu et al. use SOS at reference points to improve cluster-level spatial consistency [Wu22] (abstract).
  - Nelson et al. generate the large-scale fading and LOS state of an industrial distributed-MIMO model exactly on a grid and name SOS as the low-memory alternative [Nel26].
- Three use SOS in time with custom ACFs: a human-gesture sensing model with a damped-sinusoid ACF and N = 50, citing the fitting method [Chu24]; an underwater acoustic channel emulator [DM23] (abstract); a correlated fading model with linear complexity [WLK23] (abstract).
- Six use it for spatially consistent large-scale parameters: shadow fading in a digital twin for reinforcement-learning training [Tao25a] (abstract) and for generated user trajectories [Tao25b]; spatially correlated LSPs for THz multi-hop routing [Nav25] (abstract); shadow-fading decorrelation for moving LEO satellites [MN23] (abstract); a satellite-cellular channel model [ALLSTAR] (citation context only); D2D and satellite extensions by the authors [JRB19], [JRT22].
- The rest cite it in passing or as the reference for the QuaDRiGa simulator, mostly in machine-learning and receiver papers that use QuaDRiGa data sets [Uts22], [RU21].
- For the series: present SOS as a generic generator in 1-D to 3-D, in space and in time [Chu24], [DM23]; new phases with fixed frequencies give a new realization with the same ACF, and only the frequencies need memory [JRBT18]. Possible further tutorials: spatially consistent data sets for machine learning [Mor25], [Tao25a]; fitting a measured ACF in time [Chu24].

### 5.5 Random fields beyond radio, and dual mobility

- Methods of the same family are in use in:
  - geostatistics and hydrology: the randomization method of GSTools for conductivity and transmissivity fields on structured and unstructured meshes [GST22, Sec. 2.3, 4.2-4.3];
  - structural engineering: the spectral representation method [SD91] (abstract);
  - atmospheric optics: sparse-spectrum phase screens [Cha19];
  - ocean waves: superposition of wave components with seeded random phases in WEC-Sim [WEC];
  - material microstructures: level-cut Gaussian fields built from random waves, after Cahn and Berk [RT95];
  - machine learning: random Fourier features [RR07, Sec. 3] use the features sqrt(2)·cos(ω·x + b), which is the SOS formula; (derived) the ACF of an SOS generator is then their kernel estimate. They are also used to draw functions from Gaussian-process posteriors [WBT20].
- Common needs in these fields: evaluation at scattered points or along trajectories, unbounded and non-periodic domains, little memory, and fields that are repeatable by seed [GST22], [WBT20], [Cha19], [WEC]. FFT grids remain the choice where a dense field is needed at once: ocean rendering [Tes04], precipitation nowcasting [PYS], cosmological initial conditions [HA11] (abstract).
- Known weak spots reported there: the low frequencies of power-law spectra in phase screens [Cha19], [AOT]; too few features for large domains, called "variance starvation" in the random-feature literature [WBT20]; non-Gaussian values when amplitudes are fixed, which is the same trade-off as the equal amplitudes here. Fixed-amplitude initial conditions in cosmology are a published example of using this deliberately [AP16] (abstract).
- Dual mobility in wireless is modelled in four ways:
  - per-link updates driven by the change of link geometry [36.885, Annex A.1.4], [37.885, clause 6.2]; (derived from the code of [ns3-pl]) in ns-3 two nodes moving in parallel never change their shadowing;
  - a product joint correlation function ρ(Δtx)·ρ(Δrx), realized with a 4-D SOS over the horizontal positions of both ends [WTN06, Sec. II-IV];
  - a line integral through one shared loss field [AP09, Sec. VI-A];
  - the sum of two per-end fields [JRBT18, eq. 19].
- The sum of two per-end fields has the joint correlation (ρ_tx + ρ_rx)/2 (derived from [JRBT18, eq. 19]), not the product ρ_tx·ρ_rx that [JRBT18, eq. 18] quotes from [WTN05]. (measured with `sos.val` in the two-map mode: preset Exp256 at 20 m, 4000 independent generator pairs; moving one end 1000 m away leaves a correlation of 0.50.) The code follows eq. 19; the difference lies in the approximation of eq. 19, which [Pes20] also criticises. A tutorial on dual mobility must state this.
- TR 38.901 V19.5.0 states that mobility of one or both ends is supported [3GPP-901-19, clause 4], but its dual-mobility clause 7.6.10 changes only the Doppler term, and the ISAC clause 7.9.5.1 defines no distance between two-ended links and no spatial consistency for bistatic sensing [3GPP-901-19].
- Outside wireless (sonar and underwater communication [QS13], ocean acoustics with moving surfaces [SP08], origin-destination flows [LP08]), quantities that depend on both ends are derived from one shared field or from separate per-end effects. No explicit 6-D random field was found.
- Placement proposed: at most one sentence in part 1; its own notebook after part 2, or a long section of part 3. It needs the ACF vocabulary of part 2 and raises its own modelling questions: sum or product, one map or two, reciprocity.

### 5.6 Input for the setup section of the tutorial index

Research for the readers' setup on the tutorial index (`website_builder/content/tutorials.md`, section "Setup"); the facts are dated and have to be checked again when that text is written.


- quadriga-lib 0.12.2 on PyPI has wheels for CPython 3.10 to 3.14 on Linux (x86_64 and aarch64, manylinux with glibc 2.28 or newer) and Windows x86_64; there are no macOS, Windows-ARM or Python 3.15 wheels [PyPI-ql]. It is not on conda-forge [CF-search].
- Python 3.15 was released on 2026-10-09 [PyVer]. The Python install manager for Windows installs the latest stable Python by default [PyWin, Sec. 4.1], and the newest conda-forge `python` is 3.15.2 [CF-py]. For 3.15, pip finds no quadriga-lib wheel and tries to build from the source package (measured with `pip download`), which needs a C++ tool chain. The set-up must name the Python version (3.14 now); v0.13 should add cp315 wheels.
- Ubuntu: pip refuses to install into the system Python outside a virtual environment [PEP668], [PyPA-EME]; `python3 -m venv` fails until the package `python3-venv` is installed (read from the messages of Ubuntu 26.04's Python 3.14 [Ubu]). The apt versions on Ubuntu 24.04 are old (numpy 1.26, Notebook 6.4, no JupyterLab package) [Ubu], so the apt path is a short note at most.
- Windows: the default PowerShell execution policy on client PCs blocks the activation script `Activate.ps1` [MS-EP]; the fix given by the Python docs is `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` [PyVenv]. cmd.exe (`activate.bat`) or calling `.venv\Scripts\python.exe` directly need no change [PyVenv].
- conda: Miniforge uses only the conda-forge channel [MF]; the Anaconda `defaults` channel requires a paid licence for for-profit organisations with more than 200 employees [AnaToS, Sec. 1]. Since quadriga-lib is only on PyPI, conda installs Python, numpy, matplotlib and Jupyter, and pip installs quadriga-lib last, as the conda documentation advises for mixing [conda-pip].
- The `jupyter` package installs JupyterLab, Notebook 7, ipykernel and nbconvert [PyPI-jup], so it covers JupyterLab, Notebook, VS Code [VSC] and the refresh command of the builder README.
- Layout: scikit-learn's short per-OS blocks (create, activate, install, check) are the clearest model among the projects compared [SKL], [NumPy], [Astropy], [QuTiP], [Sionna-inst], [MNE]. venv + pip is the main path, conda the alternative. A check cell prints `sys.executable` (to catch a wrong kernel) and reads the version with `importlib.metadata`, since `quadriga_lib.version()` can end in `_AVX2` [QL-API], and suggests `%pip install --upgrade quadriga-lib` [IPy-magic] if the version is older than required.

### 5.7 Consequences for the plan

- Part 1:
  - One table of the two method families (5.1) and the three design choices, so the properties of SOS follow from its choices, with its costs named.
  - Comparison example: Cholesky on 400 random 2-D points against `sos.randn` with the `Exp256` preset at the same points (exact against scalable).
  - The ACF with the 1/e convention and a short conversion note (5.2); one Wiener-Khinchin formula ("the frequencies are samples of the spectrum"); field = ensemble, and the seed picks the realization.
  - A level-cut map with `sos.rand`, which shows a LOS state map or a two-phase microstructure [RT95].
  - Candidates: a histogram at small N against N = 256 (5.2); the scatter of the track mean over many seeds (ergodicity, 5.2).
  - Leave out: SPDE, Karhunen-Loève, sequential simulation, measure theory, Hankel transforms.
- Part 2: the step and triangle ACFs as invalid targets (5.2); ACF error against N, which also answers the ripple argument of [ns3-sgf]: inside the fitted range the error of the 256-sinusoid presets is 0.016 to 0.028 rms, beyond it the ACF keeps a residual of rms 1/sqrt(2N), 0.044 for 256 sinusoids (measured; a random rotation of the frequency set per drop is proposed in https://github.com/stjaeckel/quadriga-lib/issues/22); the slope at the origin and non-Gaussian values at small N; a travelling sea surface from custom frequencies with `sos.val`, with time as the third coordinate and the deep-water dispersion relation (measured: 256 components agree with a direct numpy evaluation within 4.5e-7; 10^6 points take about 0.4 s CPU time).
- Part 3: the four methods of 5.3, one representative tool each; a closed trajectory, where a per-link update does not return to its start value and SOS does; the ensemble bias (5.2); validation with the calibration metrics of [3GPP-901, clause 7.8.3]; which parts of clause 7.6.3 quadriga-lib implements, in the vocabulary of the TR.
- Dual mobility: placement open (5.5). The sum is filed as a defect: https://github.com/stjaeckel/quadriga-lib/issues/21 (with the measured link-pair correlations of [AP09, Table I] and the error analysis of [Pes20, Fig. 4], which the sum alone explains). The dual-mobility material waits for its fix.
- Release v0.13: wheels for Python 3.15 (5.6).

### 5.8 References

Standards

- [3GPP-901] 3GPP TR 38.901 (Rel-19), "Study on channel model for frequencies from 0.5 to 100 GHz", clauses 7.6.3.1 to 7.6.3.3 and 7.8.3. https://www.itecspec.com/3gpp/38.901/s/7.6.3.1 (rendering with garbled tables; check against the official TR).
- [3GPP-901-19] 3GPP TR 38.901 V19.5.0 (2026-09), clauses 4, 7.4.4 (exponential ACF, eq. 7.4-5), 7.5 (Table 7.5-6, correlation distances), 7.6.10, 7.9.5.1. https://www.3gpp.org/ftp/Specs/archive/38_series/38.901/38901-j50.zip
- [36.885] 3GPP TR 36.885 (Rel-14), "Study on LTE-based V2X services", Annex A.1.4. https://www.itecspec.com/3gpp/36.885/s/a.1.4
- [37.885] 3GPP TR 37.885 (Rel-15), "Study on evaluation methodology of new V2X use cases for LTE and NR", clause 6.2. https://itecspec.com/3gpp/37.885/s/6.2.2

Books and papers

- [ABB15] T. Abbas, K. Sjöberg, J. Karedal, F. Tufvesson, "A measurement based shadow fading model for vehicle-to-vehicle network simulations", arXiv:1203.3370v5, 2015. https://arxiv.org/abs/1203.3370
- [AP09] P. Agrawal, N. Patwari, "Correlated link shadow fading in multi-hop wireless networks", IEEE Trans. Wireless Commun. 8(8), 4024-4036, 2009. https://doi.org/10.1109/TWC.2009.071293
- [AP16] R. E. Angulo, A. Pontzen, "Cosmological N-body simulations with suppressed variance", MNRAS Letters, 2016. https://doi.org/10.1093/mnrasl/slw098 (abstract)
- [AT07] R. J. Adler, J. E. Taylor, Random Fields and Geometry, Springer, 2007. https://doi.org/10.1007/978-0-387-48116-6 (abstract)
- [Cha19] M. Charnotskii, "Four methods for generation of turbulent phase screens: comparison", arXiv:1911.09185, 2019. https://arxiv.org/abs/1911.09185
- [DN97] C. R. Dietrich, G. N. Newsam, "Fast and exact simulation of stationary Gaussian processes through circulant embedding of the covariance matrix", SIAM J. Sci. Comput. 18(4), 1088-1107, 1997. https://doi.org/10.1137/S1064827592240555 (content via [KB13])
- [EL06] X. Emery, C. Lantuéjoul, "TBSIM: A computer program for conditional simulation of three-dimensional Gaussian random fields via the turning bands method", Computers & Geosciences 32(10), 1615-1628, 2006. https://doi.org/10.1016/j.cageo.2006.03.001 (abstract)
- [GST22] S. Müller, L. Schüler, A. Zech, F. Heße, "GSTools v1.3: a toolbox for geostatistical modelling in Python", Geosci. Model Dev. 15, 3161-3182, 2022. https://doi.org/10.5194/gmd-15-3161-2022
- [Gud91] M. Gudmundson, "Correlation model for shadow fading in mobile radio systems", Electronics Letters 27(23), 2145-2146, 1991. https://doi.org/10.1049/el:19911328 (read; copy in the maintainer's reference collection)
- [HA11] O. Hahn, T. Abel, "Multi-scale initial conditions for cosmological simulations", MNRAS, 2011. https://doi.org/10.1111/j.1365-2966.2011.18820.x (abstract)
- [HGPC13] B. O. Hogstad, C. A. Gutiérrez, M. Pätzold, P. M. Crespo, "Classes of sum-of-cisoids processes and their statistics for the modeling and simulation of mobile fading channels", EURASIP J. Wireless Commun. Netw. 2013:125. https://doi.org/10.1186/1687-1499-2013-125 (abstract)
- [JRBT18] S. Jaeckel, L. Raschkowski, F. Burkhardt, L. Thiele, "Efficient sum-of-sinusoids-based spatial consistency for the 3GPP new-radio channel model", IEEE Globecom Workshops, 2018. https://doi.org/10.1109/GLOCOMW.2018.8644265; full text on this website: `website_builder/static/publications/2018_sum_of_sinusoids.md`
- [KB13] D. P. Kroese, Z. I. Botev, "Spatial process generation", arXiv:1308.0399, 2013. https://arxiv.org/abs/1308.0399
- [Kra70] R. H. Kraichnan, "Diffusion by a random velocity field", Phys. Fluids 13, 22-31, 1970. https://doi.org/10.1063/1.1692799 (content via [GST22])
- [LP08] J. LeSage, R. K. Pace, "Spatial econometric modeling of origin-destination flows", J. Regional Science 48(5), 941-967, 2008. https://doi.org/10.1111/j.1467-9787.2008.00573.x (bibliographic)
- [LRL11] F. Lindgren, H. Rue, J. Lindström, "An explicit link between Gaussian fields and Gaussian Markov random fields: the stochastic partial differential equation approach", J. R. Stat. Soc. B 73(4), 423-498, 2011. https://doi.org/10.1111/j.1467-9868.2011.00777.x
- [Man98] J. Mann, "Wind field simulation", Probabilistic Engineering Mechanics 13(4), 269-282, 1998. https://doi.org/10.1016/S0266-8920(97)00036-2 (abstract)
- [MW82] A. Mantoglou, J. L. Wilson, "The turning bands method for simulation of random fields using line generation by a spectral method", Water Resour. Res. 18(5), 1379-1394, 1982. https://doi.org/10.1029/WR018i005p01379 (abstract)
- [NYU19] S. Ju, O. Kanhere, Y. Xing, T. S. Rappaport, "A millimeter-wave channel simulator NYUSIM with spatial consistency and human blockage", IEEE GLOBECOM 2019. https://arxiv.org/abs/1908.09762
- [Pae04] M. Pätzold, "On the stationarity and ergodicity of fading channel simulators based on Rice's sum-of-sinusoids", Int. J. Wireless Inf. Netw. 11(2), 63-69, 2004. https://doi.org/10.1023/B:IJWI.0000034538.27413.60 (bibliographic)
- [Pae11] M. Pätzold, Mobile Radio Channels, 2nd ed., Wiley, 2011. https://doi.org/10.1002/9781119974116 (bibliographic)
- [PKL96] M. Pätzold, U. Killat, F. Laue, "A deterministic digital simulation model for Suzuki processes with application to a shadowed Rayleigh land mobile radio channel", IEEE Trans. Veh. Technol. 45(2), 318-331, 1996. https://doi.org/10.1109/25.492906 (bibliographic)
- [QS13] P. Qarabaqi, M. Stojanovic, "Statistical characterization and computationally efficient modeling of a class of underwater acoustic communication channels", IEEE J. Oceanic Eng. 38(4), 701-717, 2013. (bibliographic)
- [Ric44] S. O. Rice, "Mathematical analysis of random noise", Bell Syst. Tech. J. 23(3), 282-332, 1944. https://doi.org/10.1002/j.1538-7305.1944.tb00874.x
- [Ric45] S. O. Rice, "Mathematical analysis of random noise" (Parts III and IV), Bell Syst. Tech. J. 24(1), 46-156, 1945. https://doi.org/10.1002/j.1538-7305.1945.tb00453.x (read from a scan: Sec. 3.3, expected number of zeros, eq. 3.3-11, p. 54; infinite for the exponential ACF, p. 55)
- [RR07] A. Rahimi, B. Recht, "Random features for large-scale kernel machines", NIPS 20, 2007. https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html
- [RT95] A. P. Roberts, M. Teubner, "Transport properties of heterogeneous materials derived from Gaussian random fields: bounds and simulation", Phys. Rev. E 51(5), 4141-4154, 1995. https://doi.org/10.1103/PhysRevE.51.4141
- [RW06] C. E. Rasmussen, C. K. I. Williams, Gaussian Processes for Machine Learning, MIT Press, 2006. http://gaussianprocess.org/gpml/chapters/ (chapters 2, 4 and appendix A)
- [SD91] M. Shinozuka, G. Deodatis, "Simulation of stochastic processes by spectral representation", Appl. Mech. Rev. 44(4), 191-204, 1991. https://doi.org/10.1115/1.3119501 (abstract)
- [SD96] M. Shinozuka, G. Deodatis, "Simulation of multi-dimensional Gaussian stochastic fields by spectral representation", Appl. Mech. Rev. 49(1), 29-53, 1996. https://doi.org/10.1115/1.3101883 (abstract)
- [SDH87] M. Shinozuka, G. Deodatis, T. Harada, "Digital simulation of seismic ground motion", Technical Report NCEER-87-0017, 1987. https://nehrpsearch.nist.gov/static/files/NSF/PB88155197.pdf
- [SH09] P. Schneider, J. Hartlap, "Constrained correlation functions", Astronomy & Astrophysics 504(3), 705-717, 2009. https://doi.org/10.1051/0004-6361/200912424 (read: arXiv:0905.0577v1; bounds in Sec. 2, eqs. 5-6)
- [SN26] C. Safta, H. N. Najm, "Numerical considerations for the construction of Karhunen-Loève expansions", arXiv:2603.19108, 2026. https://arxiv.org/abs/2603.19108
- [SP08] M. Siderius, M. B. Porter, "Modeling broadband ocean acoustic transmissions with time-varying sea surfaces", J. Acoust. Soc. Am. 124, 137-150, 2008. https://doi.org/10.1121/1.2920959 (bibliographic)
- [Tes04] J. Tessendorf, "Simulating ocean water", SIGGRAPH course notes, 1999-2004. https://jtessen.people.clemson.edu/reports/papers_files/coursenotes2004.pdf
- [Van10] E. Vanmarcke, Random Fields: Analysis and Synthesis, revised ed., World Scientific, 2010. https://doi.org/10.1142/5807 (bibliographic; variance function via [WW07])
- [Vee88] P. S. Veers, "Three-dimensional wind simulation", Sandia report SAND88-0152, 1988. https://www.osti.gov/biblio/7102613 (abstract)
- [VieSLS] M. K. Müller et al., "Flexible multi-node simulation of cellular mobile communications: the Vienna 5G System Level Simulator", EURASIP J. Wireless Commun. Netw. 2018:227. https://doi.org/10.1186/s13638-018-1238-7
- [VieWSA] T. Dittrich, M. Taranetz, M. Rupp, "An efficient method for avoiding shadow fading maps in system level simulations", WSA 2017. https://www.tib.eu/en/search/id/vde:454394070 (abstract)
- [WBT20] J. T. Wilson, V. Borovitskiy, A. Terenin, P. Mostowsky, M. P. Deisenroth, "Efficiently sampling functions from Gaussian process posteriors", ICML 2020, PMLR 119, 10292-10302. https://proceedings.mlr.press/v119/wilson20a.html
- [WTN05] Z. Wang, E. K. Tameh, A. R. Nix, "A sum-of-sinusoids based simulation model for the joint shadowing process in urban peer-to-peer radio channels", IEEE VTC 2005-Fall, vol. 3, pp. 1732-1736. https://doi.org/10.1109/VETECF.2005.1558401 (read; eq. 4 is the product joint correlation; its ACF uses a := ln(2)/d_corr, i.e. 1/2 at d_corr)
- [WTN06] Z. Wang, E. K. Tameh, A. Nix, "Simulating correlated shadowing in mobile multihop relay/ad-hoc networks", IEEE C802.16j-06/060, 2006. https://www.ieee802.org/16/relay/contrib/C80216j-06_060.pdf
- [WW07] Wang Ying-he, Wang Bao-tian, "Study on autocorrelation model and reduction function of variance of soil random field", Proc. ISGSR 2007. https://www.geoengineer.org/geosnet/ISGSR2007/Part1Paper8.pdf

Papers citing [JRBT18]

- [ALLSTAR] J. Kim et al., "5G-ALLSTAR Deliverable D3.1: Spectrum usage analysis and channel model", 2019. https://5g-allstar.eu/wp-content/uploads/2019/07/815323_Deliverable_3.1_Spectrum-usage-analysis-and-channel-model.pdf (citation context only)
- [Chu24] J. Chuang, R. Caromi, J. Senic et al., "Quasi-deterministic channel propagation model for human sensing: gesture recognition use case", IEEE Open J. Antennas Propag. 5(3), 2024. https://doi.org/10.1109/OJAP.2024.3371834
- [DM23] I. Dey, N. Marchetti, "Channel emulator framework for underwater acoustic communications", Applied Sciences 13(9), 5818, 2023. https://doi.org/10.3390/app13095818 (abstract)
- [Hua23] H. Huang, K. Wang, P. Wu et al., "Flexible 3GPP MIMO channel modeling and calibration with spatial consistency", IEEE Access, 2023. https://doi.org/10.1109/ACCESS.2023.3304247 (abstract)
- [JRB19] S. Jaeckel, L. Raschkowski, F. Burkhardt, "A spatially consistent geometric D2D small-scale fading model for multiple frequencies", IEEE VTC-Fall 2019. https://doi.org/10.1109/VTCFall.2019.8891407
- [JRT22] S. Jaeckel, L. Raschkowski, L. Thiele, "A 5G-NR satellite extension for the QuaDRiGa channel model", EuCNC/6G Summit 2022. https://doi.org/10.1109/EuCNC/6GSummit54941.2022.9815679
- [MN23] M. Momani, M. Nabeel, "Towards realistic stochastic channel modeling for mobility-enabled low earth orbit satellites", IEEE CSCN 2023. https://doi.org/10.1109/CSCN60443.2023.10453182 (abstract)
- [Mor25] J. Morais, A. Malhotra, S. Hamidi-Rad et al., "Comparing stochastic and ray-tracing datasets in machine learning for wireless applications", Asilomar 2025. https://doi.org/10.1109/IEEECONF67917.2025.11443737
- [Nav25] P. Navade, I. F. Akyildiz, C. Han, "Routing based on wireless channel effects for multi-hop outdoor communication in the THz band", IEEE Trans. Commun., 2025. https://doi.org/10.1109/TCOMM.2025.3615773 (abstract)
- [Nel26] C. Nelson, S. Willhammar, F. Tufvesson, "A measurement-based spatially consistent channel model for distributed MIMO in industrial environments", IEEE Trans. Wireless Commun., 2026. https://doi.org/10.1109/TWC.2026.3728836
- [Pes20] A. M. Pessoa, C. F. M. e Silva, I. M. Guerreiro, "A spatially consistent Gaussian process for dual mobility in the 3D space", IEEE Wireless Commun. Lett., 2020. https://doi.org/10.1109/LWC.2020.2992725 (abstract)
- [Pes21] A. M. Pessoa, I. M. Guerreiro, C. F. M. e Silva, "A positive semidefinite autocorrelation function for modeling 3D Gaussian processes", IEEE Trans. Veh. Technol., 2021. https://doi.org/10.1109/TVT.2021.3055040 (abstract)
- [RU21] V. Rizzello, W. Utschick, "Learning the CSI denoising and feedback without supervision", IEEE SPAWC 2021. https://doi.org/10.1109/SPAWC51858.2021.9593213
- [Tao25a] Z. Tao, W. Xu, X. You, "Large vision model-enhanced digital twin with deep reinforcement learning for user association and load balancing in dynamic wireless networks", IEEE J. Sel. Areas Commun., 2025. https://doi.org/10.1109/JSAC.2025.3644284 (abstract)
- [Tao25b] Z. Tao, W. Xu, X. You, "Map2Traj: street map piloted zero-shot trajectory generation method for wireless network optimization", IJCAI 2025. https://doi.org/10.24963/ijcai.2025/1041
- [Uts22] W. Utschick, V. Rizzello, M. Joham et al., "Learning the CSI recovery in FDD systems", IEEE Trans. Wireless Commun., 2022. https://doi.org/10.1109/TWC.2022.3149946
- [WLK23] T. W. Weber, E. Lyczkowski, W. Kieß, "Non-geometric correlated channel fading model with linear complexity", EuCNC/6G Summit 2023. https://doi.org/10.1109/EuCNC/6GSummit58263.2023.10188367 (abstract)
- [Wu22] C. Wu, Y. Zhu, W. Wang et al., "Improvement of the cluster-level spatial consistency of channel simulator with reference points transition method", IEEE Trans. Veh. Technol., 2022. https://doi.org/10.1109/TVT.2022.3161728 (abstract)

Citation databases (queried 2026-11-05)

- [Crossref] https://api.crossref.org/works/10.1109/glocomw.2018.8644265
- [OpenAlex] https://api.openalex.org/works?filter=cites:W2885343185
- [S2] Semantic Scholar Graph API, citations with contexts: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/GLOCOMW.2018.8644265/citations
- [OCit] OpenCitations Index API v2: https://api.opencitations.net/index/v2/citations/doi:10.1109/glocomw.2018.8644265

Software sources and documentation

- [air] 5G-air-simulator, `src/channel/propagation-model/channel-realization.cpp`, commit 7ac9816. https://github.com/telematics-lab/5G-air-simulator/blob/7ac9816c9409d33567ffb5f2130a7ea6e43e55f6/src/channel/propagation-model/channel-realization.cpp
- [AOT] AOtools documentation, "Atmospheric Turbulence". https://aotools.readthedocs.io/en/latest/turbulence.html
- [AOT-src] AOtools, `aotools/turbulence/phasescreen.py`. https://github.com/AOtools/aotools/blob/main/aotools/turbulence/phasescreen.py
- [GST] GSTools documentation, `gstools.covmodel.Gaussian`. https://geostat-framework.readthedocs.io/projects/gstools/en/stable/api/gstools.covmodel.Gaussian.html
- [GST-src] GSTools v1.7.0, `src/gstools/field/generator.py` (`RandMeth`). https://github.com/GeoStat-Framework/GSTools/blob/v1.7.0/src/gstools/field/generator.py
- [gstat] gstat 2.1-7 documentation, `krige`. https://r-spatial.github.io/gstat/reference/krige.html
- [Hip] DTU Wind, Hipersim documentation, "MannTurbulenceField", v0.1.22. https://hipersim.pages.windenergy.dtu.dk/hipersim/MannTurbulenceField.html
- [MW-cdl] MathWorks, "nrCDLChannel", 5G Toolbox R2026b. https://www.mathworks.com/help/5g/ref/nrcdlchannel-system-object.html
- [MW-h] MathWorks, `h38901Channel.m` example helper (`generateLSP`, `getGridResolution`, `mobilityManager`), commit 714e513c. https://github.com/mathworks/wireless-network-modeling-workshop/blob/714e513c3716116060c18339ec30aeb03d189457/h38901Channel.m
- [ns3-ch] ns-3.49, `src/spectrum/model/three-gpp-channel-model.cc`. https://gitlab.com/nsnam/ns-3-dev/-/blob/ns-3.49/src/spectrum/model/three-gpp-channel-model.cc
- [ns3-doc] ns-3.49, `src/spectrum/doc/spectrum.rst`. https://gitlab.com/nsnam/ns-3-dev/-/blob/ns-3.49/src/spectrum/doc/spectrum.rst
- [ns3-pl] ns-3.49, `src/propagation/model/three-gpp-propagation-loss-model.cc` (`GetShadowing`). https://gitlab.com/nsnam/ns-3-dev/-/blob/ns-3.49/src/propagation/model/three-gpp-propagation-loss-model.cc
- [ns3-rel] ns-3.49, `RELEASE_NOTES.md` (3.47 and 3.49 entries). https://gitlab.com/nsnam/ns-3-dev/-/blob/ns-3.49/RELEASE_NOTES.md
- [ns3-sgf] ns-3.49, `src/propagation/model/spatial-gaussian-field.h` and `.cc`. https://gitlab.com/nsnam/ns-3-dev/-/blob/ns-3.49/src/propagation/model/spatial-gaussian-field.h
- [OAI-doc] OpenAirInterface, `openair1/SIMULATION/TOOLS/DOC/channel_simulation.md`, commit 88245ba8. https://github.com/OPENAIRINTERFACE/openairinterface5g/blob/88245ba846bd046720eb3503e17c297ec80c1606/openair1/SIMULATION/TOOLS/DOC/channel_simulation.md
- [OF-SS] OpenFAST documentation, SeaState input files. https://openfast.readthedocs.io/en/main/source/user/seastate/input_files.html
- [OF-TS] OpenFAST, `modules/turbsim/src/TSsubs.f90`. https://github.com/OpenFAST/openfast/blob/main/modules/turbsim/src/TSsubs.f90
- [PYS] pysteps v1.4 documentation, `pysteps.noise`. https://pysteps.readthedocs.io/en/v1.4/pysteps_reference/noise.html
- [QL-API] quadriga-lib Python API documentation, `sos` functions and `version`: `docs/quadriga_lib_api_python.md` in this repository.
- [Simu5G] Simu5G v1.8.0, `src/simu5g/stack/phy/channelmodel/StochasticChannelModel.cc` (`computeShadowing`). https://github.com/Unipisa/Simu5G/blob/v1.8.0/src/simu5g/stack/phy/channelmodel/StochasticChannelModel.cc
- [Sio-cc] Sionna v2.2.0, `src/sionna/phy/channel/tr38901/channel_coefficients.py`. https://github.com/NVlabs/sionna/blob/v2.2.0/src/sionna/phy/channel/tr38901/channel_coefficients.py
- [Sio-lsp] Sionna v2.2.0, `src/sionna/phy/channel/tr38901/lsp.py`. https://github.com/NVlabs/sionna/blob/v2.2.0/src/sionna/phy/channel/tr38901/lsp.py
- [Sio-rays] Sionna v2.2.0, `src/sionna/phy/channel/tr38901/rays.py`. https://github.com/NVlabs/sionna/blob/v2.2.0/src/sionna/phy/channel/tr38901/rays.py
- [Sio-rel] Sionna release notes v2.0.0 to v2.2.0. https://github.com/NVlabs/sionna/releases
- [Sio-sc] Sionna v2.2.0, `src/sionna/phy/channel/tr38901/spatial_consistency.py`. https://github.com/NVlabs/sionna/blob/v2.2.0/src/sionna/phy/channel/tr38901/spatial_consistency.py
- [Sio-slc] Sionna v2.2.0, `src/sionna/phy/channel/tr38901/system_level_channel.py`. https://github.com/NVlabs/sionna/blob/v2.2.0/src/sionna/phy/channel/tr38901/system_level_channel.py
- [V2X] WiLabV2Xsim, `MatFilesSINR/computeChannelGain.m`, commit 612bf26. https://github.com/V2Xgithub/WiLabV2Xsim/blob/612bf26ab91f294b635263065c26216921ad07ad/MatFilesSINR/computeChannelGain.m
- [VieLLS] TU Wien, "Vienna 5G Link Level Simulator v1.3 User Manual", 2022. https://www.tuwien.at/etit/tc/wp-content/uploads/2022/10/5gll-usermanual.pdf
- [WEC] WEC-Sim documentation, "Advanced Features". https://wec-sim.github.io/WEC-Sim/main/user/advanced_features.html
- [WIN] P. Kyösti, `LScorrelation.m`, WINNER II / IMT-A MATLAB code, copy in wiless/M2135, commit bde5dd9. https://github.com/wiless/M2135/blob/bde5dd9b49f9768f454a270fd42f9766e4ef8d04/LScorrelation.m

Set-up sources

- [AnaToS] Anaconda, Inc., "Terms of Service", July 15, 2025. https://www.anaconda.com/legal/terms/terms-of-service
- [Astropy] Astropy, "Installation". https://docs.astropy.org/en/stable/install.html
- [CF-py] Anaconda.org API, conda-forge `python`. https://api.anaconda.org/package/conda-forge/python
- [CF-search] conda-forge: no `quadriga-lib` package or feedstock (https://api.anaconda.org/package/conda-forge/quadriga-lib returns 404; queried 2026-11-05).
- [conda-pip] conda, "Managing environments", section "Using pip in an environment". https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
- [IPy-magic] IPython, "Built-in magic commands" (`%pip`). https://ipython.readthedocs.io/en/stable/interactive/magics.html
- [MF] conda-forge, Miniforge README. https://github.com/conda-forge/miniforge
- [MNE] MNE-Python, "Installing MNE-Python manually". https://mne.tools/stable/install/manual_install.html
- [MS-EP] Microsoft, "about_Execution_Policies". https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies
- [NumPy] NumPy, "Installing NumPy". https://numpy.org/install/
- [PEP668] PEP 668, "Marking Python base environments as 'externally managed'". https://peps.python.org/pep-0668/
- [PyPA-EME] PyPA, "Externally Managed Environments". https://packaging.python.org/en/latest/specifications/externally-managed-environments/
- [PyPI-jup] PyPI, `jupyter` 1.1.1 metadata. https://pypi.org/pypi/jupyter/json
- [PyPI-ql] PyPI, `quadriga-lib` metadata and files. https://pypi.org/pypi/quadriga-lib/json (queried 2026-11-05)
- [PyVenv] Python documentation, "venv — Creation of virtual environments". https://docs.python.org/3/library/venv.html
- [PyVer] Python Developer's Guide, "Status of Python versions". https://devguide.python.org/versions/
- [PyWin] Python documentation, "Using Python on Windows". https://docs.python.org/3/using/windows.html
- [QuTiP] QuTiP, "Installation". https://qutip.readthedocs.io/en/latest/installation.html
- [Sionna-inst] NVIDIA, Sionna 2.2.0, "Installation". https://nvlabs.github.io/sionna/installation.html
- [SKL] scikit-learn, "Installing scikit-learn". https://scikit-learn.org/stable/install.html
- [Ubu] Ubuntu package data: `/usr/lib/python3.14/EXTERNALLY-MANAGED` and the venv module of Ubuntu 26.04; package versions from https://packages.ubuntu.com (noble).
- [VSC] Microsoft, "Jupyter Notebooks in VS Code". https://code.visualstudio.com/docs/datascience/jupyter-notebooks

## 6. Part 1: scope and content (sketch)

First sketch, for discussion; nothing is written yet. Notebook `tutorials/sos_01_random_fields.ipynb`, Python. Metadata: title "Random fields and the sum-of-sinusoids method", level `beginner`, `order` 1, `duration_min` about 25. The notebook runs in well under a minute (all calls together took 0.17 s CPU time in the environment check, section 5.6). Each section below is one `##` heading; code cells are short (a handful of lines), use numpy and matplotlib only, and every figure gets an alt text.

### 6.1 Sections

1. Requirements. One Markdown line: quadriga-lib v0.13 or newer, set-up as on the tutorial index (`website_builder/content/tutorials.md`). Possibly a short version check (see 6.3).
2. What a random field is. A random function of position; one realization against the ensemble; stationarity and isotropy in one sentence each [KB13, Sec. 1], [RW06, Sec. 4.1]. One sentence per field where they occur: shadow fading and the large-scale parameters of channel models, soil and aquifer properties [GST22], ocean surfaces [Tes04], turbulence phase screens [Cha19], material microstructures [RT95]. No code.
3. The autocorrelation function. Definition in words and one formula; the decorrelation distance with the 1/e convention and a note on the other conventions (5.2); smooth (Gauss) against rough (Exp) fields; one Wiener-Khinchin sentence: the ACF and the power spectrum are a Fourier pair [RW06, Sec. 4.2.1, eq. 4.6]. Code: the target ACFs of the `Exp`, `Gauss` and `Comb` presets from `sos.preset`. Figure 1: three ACF curves over the distance.
4. How random fields are generated. The table of the two families (matrix methods, sums of waves) and the three design choices of the spectral family (5.1), with the place of the SOS method in it. No code.
5. The sum-of-sinusoids method. `s(x) = sqrt(2/N) · sum_n cos(2π f_n·x + φ_n)`: the frequencies are samples of the spectrum and set the ACF, the phases select the realization, the central limit theorem makes the values nearly Gaussian [Ric44], [JRBT18]. Strengths and costs in two short lists, including the residual of 1/sqrt(2N) beyond the fitted range (issue #22). No code, or one cell that prints the size of a preset (`freq.shape`).
6. A first field. `sos.randn` along a 100 m track for one decorrelation distance, `Exp256` against `Gauss256`: rough against smooth. Then `sos.map` on a 2-D grid, the two presets side by side, as in Fig. 1 of [JRBT18]. Figures 2 (two track curves) and 3 (two maps).
7. Choosing a preset. The table of presets (`Exp`, `Gauss`, `Comb`; 128 to 1024 sinusoids), the accuracy of each step (3 to 4 dB per doubling, from the API documentation) and the cost (time ∝ N); `Comb256` as the default and why. No figure, or the approximation error against N from `sos.calc_mse` (one short loop).
8. Checking the correlation. The ACF estimated from generated values (`sos.acf_estimate`) against the target, 1-D, as Fig. 4 (upper) of [JRBT18]; optionally the 2-D target (`sos.acf_2d`) against the ACF of the generator (`sos.acf_approx`), as Fig. 2 of [JRBT18]. Figure 4 (and 5).
9. Repeatable simulations. The seed selects the phases: the same seed gives the same field at any later position (extend the track of section 6 and show that the old part is unchanged). A seed selects phases, not a field: the same seed with another decorrelation distance gives a related field, so independent quantities need one call with several decorrelation distances or different seeds (API documentation). Keeping a generator: `sos.preset` + `sos.init_phases` + `sos.val`. Figure 6: the extended track.
10. Uniform and integer values. `sos.rand` and a threshold: a map of two states (LOS/NLOS, or two material phases [RT95]); `sos.randi`: a map of integer classes. Figure 7.
11. Comparison with a matrix method. Cholesky of an exponential covariance matrix on 400 random 2-D points against `sos.randn` with `Exp256` at the same points (5.1, 5.7): both give correlated values; Cholesky is exact but needs the n × n matrix and a new factorisation for new points, the SOS generator evaluates any new point. Figure 8: the two point sets side by side.
12. Summary and further reading. [RW06, ch. 4]; [Pae11] for wireless readers, [SD91] for others; what part 2 covers.

### 6.2 Left out of part 1

SPDE, Karhunen-Loève, sequential simulation, measure theory, Hankel transforms (5.7); dual mobility (issue #21, at most one sentence); bias removal (part 3); custom ACFs and the fitting figures of [JRBT18] (part 2).

### 6.3 Decisions on the sketch

1. Order: result first and interleaved. A map right after the introduction (one cell of about four lines with `sos.map`) shows what the notebook builds; each concept then comes with a small cell, including the sum-of-sinusoids formula evaluated by hand in numpy and compared with `sos.val`.
2. Demos: the scatter of the track mean over many seeds goes into section 9 (one realization is not the ensemble; leads to the bias of part 3). The histogram at small N goes into the hand-written sum-of-sinusoids cell (a subset of the preset frequencies with `sos.val`).
3. Preset accuracy: a table only, in correlation units (rms error inside the fitted range, the floor 1/sqrt(2N) beyond it); the curve of the error against N goes to part 2.
4. The 2-D ACF figure (target from `sos.acf_2d` against the generator from `sos.acf_approx`) is included, one figure with two panels: it shows the isotropy that separable grid filters lack (5.3).

The version cell, the handling of outputs and the figure format follow `tutorials/README.md`.

## 7. Part 2: scope, decisions and findings

Notebook `tutorials/sos_02_custom_acf.ipynb`, Python. Metadata: title "Custom correlation functions and the limits of the sum-of-sinusoids method", level `advanced`, `order` 11, `duration_min` 35. The notebook runs in about 30 s (all fits included). Numbers below are printed by the notebook unless marked otherwise; "(measured)" marks checks run for the plan with a local build of quadriga-lib (main branch).

### 7.1 Sections

1. A field with two length scales: 0.6·exp(−d/5 m) + 0.4·exp(−d/50 m), fitted in 3-D with 256 sinusoids on 201 geometrically spaced sample points up to 250 m, `freq_min` 0.001 1/m, `freq_max` 1 1/m, 20 test directions; a map.
2. The frequencies are the free parameters: equal amplitudes, so the density of the frequencies takes the place of the spectrum; a histogram of |f|; what `sos.generate` does [JRBT18, Sec. III] and what is the library's (regular directions on the half circle or half sphere, fit along the sinusoid's own direction); its arguments.
3. Judging a fit: the same target with the default settings loses the short scale (ACF 0.99 at 1 m against 0.88; largest error of the direction-mean ACF 0.23 against 0.014) while the scores of `sos.calc_mse` differ by at most 1.8 dB; the ACF along 200 directions; the error map from `sos.acf_approx` minus `sos.acf_2d`; `acf_tail` and `freq_min`; `sos.acf_estimate` over 20 realizations.
4. Using your own frequencies: `sos.init_phases`, `sos.val`, `sos.map`; division of the frequencies scales the distances [JRBT18, Sec. III]; scaling one column makes the field anisotropic.
5. Targets that are not ACFs: Bochner [RW06, Sec. 4.2.1, Thm. 4.1]; the step with the three-point matrix (determinant −1); fits with 16, 64 and 256 sinusoids reach the same rms error (0.186, 0.157, 0.153); the damped cosine exp(−d/20 m)·cos(2πd/10 m), the form of [Chu24, Sec. III-B, eq. 2], fitted in 1-D (rms 0.020 with 32 sinusoids) and in 2-D and 3-D, where the fits stop at the bounds −0.402 and −0.217 [SH09, Sec. 2, eqs. 5-6]; the 2-D fit puts 80 % of its frequencies on a ring at 0.104 to 0.106 1/m (ACF J0); the triangle in one sentence (2-D fit rms 0.026).
6. How many sinusoids: own 3-D fits of the exponential ACF with 8 to 256 sinusoids and the presets `Exp128` to `Exp1024`; error inside the fitted range and between 5 and 30 d0 against 1/sqrt(2N); 3 dB per doubling [JRBT18, Sec. V, Fig. 3]; a fit with 300 test directions reaches the preset (0.028 against 0.028 at 256); a fit without `freq_min` keeps a mean of +0.015 beyond the range; the residual belongs to the table; its average over all rotations, (1/N)·Σ sin(2π|f|d)/(2π|f|d), has an rms of 0.0053 for `Exp256` and decays with distance; 1-D fits leave the tail uncontrolled (damped cosine, 32 sinusoids: rms 0.209 between 50 and 1000 m against 0.125).
7. The slope at the origin: the `Exp` presets near d = 0; zero crossings per 100 m from 2·sqrt(mean f_x²) [Ric45, Sec. 3.3, eq. 3.3-11] and counted on a 10 km track (Exp128 7.5 to Exp1024 11.1, Comb256 4.6; scatter between seeds ±0.2 to ±0.3).
8. Few sinusoids: tail probabilities against N (no value below −3 with 4 sinusoids, about half of the Normal share with 8, about 90 % with 32).
9. A traveling sea surface: wave vectors drawn from the Phillips spectrum [Tes04, eqs. 40, 41] with equal amplitudes (Tessendorf uses an FFT lattice with Gaussian amplitudes, eqs. 36, 42), deep-water dispersion [Tes04, eq. 31], time as the third coordinate of `sos.map`; the principle of random Fourier features [RR07, Sec. 3]; two frames 2 s apart.
10. Limits of the method, summary, references.

### 7.2 Decisions

- Main example: the two-scale ACF, which teaches the spacing of the sample points and the frequency limits; the damped cosine serves as the example of a target that is valid in 1-D only.
- The triangle gets one sentence: its 2-D spectrum is only about 2 % negative, and a 2-D fit with 256 sinusoids follows it (rms 0.026), so it does not show the dependence on the dimension.
- The residual 1/sqrt(2N) is presented for a fixed table of frequencies, with the average over rotations as what a random rotation per realization would give; no recipe for rotating tables and no issue link (https://github.com/stjaeckel/quadriga-lib/issues/22 is open; to be revisited when it is resolved).
- The ripple argument of [ns3-sgf] is answered with the numbers only; ns-3 is not named.
- The sea surface has its own section before the limits; keeping only the waves that move with the wind is a choice of the notebook and is said so.
- Error against N: own fits and presets in correlation units (rms), with one line in dB from `sos.calc_mse`.
- Left out: dual mobility (https://github.com/stjaeckel/quadriga-lib/issues/21), `sos.remove_bias` and the ensemble bias (part 3), fits with 1024 sinusoids, a second histogram of the values, conditioning on measured values.

### 7.3 Findings

- Fit settings: each sample point counts equally in the error of `sos.generate`, so the spacing of the sample points weights the fit; the default upper limit 15/max(dist) limits fine scales when the range is long. For the two-scale target, either change alone does not help (measured: largest error 0.17 to 0.31); both together reduce it from 0.20 to 0.018 (measured, 2-D). Raising `freq_max` alone for exp(−d/5 m) fitted to 250 m moves the ACF at 5 m from 0.638 to 0.486 (target 0.368; measured, 2-D).
- Cost of `sos.generate` (measured): 2-D and 3-D fits take up to about 1 s for 256 sinusoids on 200 sample points with the default limits and 20 to 30 test directions, about 2 s with 300; the two-scale fit with `freq_max` 1 1/m over 250 m takes about 5 s; 1-D fits run up to 1000 passes and take several seconds.
- 1-D fits do not decay to 1/sqrt(2N) beyond the fitted range: the fit stacks sinusoids closer together than it can resolve over the range; `freq_min` turns the plateau of near-zero frequencies into an oscillation of about the same size (measured: exponential ACF, 256 sinusoids, rms beyond the range 0.133, with `freq_min` 0.247). A tail term in the cost of the fit reduces it to 0.018 at an in-range error of 0.0053 instead of 0.0049 (measured in a numpy prototype). Filed as https://github.com/stjaeckel/quadriga-lib/issues/24; the 1-D statements of part 2 are to be revisited when it is resolved.
- Frequencies drawn from a spectrum (sea surface) give the target ACF only on average over draws; one draw has an error of about 1/sqrt(2N) at all distances, more than a fit inside its range.
- Reference check of part 2: the zero-crossing formula is in Parts III and IV of Rice [Ric45], not in [Ric44]; Chuang et al. model the residual fluctuations of the multipath parameters with the damped cosine, not the channel of moving hands and arms; GPML gives the triangle only as the 1-D member of eq. 4.21; Tessendorf's crests can be sharpened ([Tes04, Sec. 4.6]). The notebook follows these readings.
