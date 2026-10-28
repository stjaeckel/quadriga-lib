# SSF literature scan: spatial consistency, multi-frequency and D2D for TR 38.901 steps 5–7

Scan date 2026-10-28. Labels: **[V]** verified in a primary source I read; **[S]** from a secondary source only
(abstract, search snippet, blog); **[I]** my inference. 3GPP Tdoc zip files were **not downloaded or opened**
(per the brief), so the content of every R1 contribution below is "not read". What is verified are the CR metadata
(3GPP portal CR lists) and the TR text before and after each CR (ETSI PDFs of TR 38.900 v14.2.0, TR 38.901 v14.0.0
and v14.3.0, plus the local v19.5.0 markdown).

## 1. Summary

- **Nothing I found reconciles SC + MF + D2D for the 38.901 SSF in one method, except Jaeckel et al. 2019
  (QuaDRiGa 2.2).** No later paper, simulator or 3GPP contribution does it [V/S].
- **The TR partly reconciles SC and MF, but only for Procedure B.** Since v14.0.0 (2017-03), 7.6.3.2 says that in MF
  simulations the *largest* spread over all frequencies is used in the uniform delay and angle ranges (7.6-15/16),
  so delays and angles are the same for all frequencies and only the powers differ. The sentence is still in v19.5
  (line 2739 of the local md) [V]. The baseline 7.5 path and 7.6.5.1 have no such rule.
- **7.6.5.1 came from an Ericsson + Fraunhofer HHI CR at RAN1#88 (Feb 2017).** It was absent in 38.900 v14.2.0 and
  present in 38.901 v14.0.0 [V]. The most likely source is CR 0078, R1-1703647 [I, strong]. The stated reason is
  the Note in 7.6.5: under 7.5, delays and angles become frequency-dependent whenever DS or AS depends on frequency [V].
- **Procedure A and Procedure B were already in TR 38.900 v14.0.0 (June 2016).** I could not identify the first
  proposers; the RAN1#84bis/#85 WFs were not found. In Rel-14, Procedure B was reworded by CR 0083 (Ericsson, Nokia,
  RAN1#88) [V metadata, I content]. Procedure A was rewritten by CR 0015 (ZTE, DOCOMO) and CR 0019 (HHI) in 2017
  [V].
- **Rel-19 (7–24 GHz, ISAC) added a "link-correlated" SC type for UT-UT links and a cluster-number range (7.6.15).**
  It did not change steps 5–7, did not define a dual-position random process, and has no reciprocity rule [V].
- **Deviation from baseline statistics:**
  - The only quantitative input-vs-output study is in the Jaeckel 2019 paper itself. It found a good match for DS,
    ASD and ESA, and offsets for ASA (capped by K) and ESD (about doubled).
  - I found no published comparison of baseline 7.5 output spreads against Procedure B, 7.6.5.1 or the paper method.
  - 3GPP calibrated SC only with correlation and "varying-rate" metrics (Table 7.8-5), not with spread fidelity [V].
  - The library will have to produce this comparison itself.

## 2. 3GPP contributions

| Doc | Meeting | Company | Topic | Finding | Label |
|---|---|---|---|---|---|
| TR 38.900 v14.0.0 | RAN#72, 2016-06 | (study output) | First TR with 7.6.3 SC, Proc A/B, 7.6.5 MF | Features predate all CRs below. Originating WFs (RAN1#84bis/#85) not identified. | V (v14.2.0 read; v14.0.0 content inferred from CR list) |
| R1-167236 (CR 0008) | RAN1#86, 2016-08 | Huawei, HiSilicon | Correction to SC 7.6.3 | Merged | V metadata; not read |
| R1-168478 (CR 0042) | RAN1#86 | Samsung | Clarification of spatially consistent random variables | Agreed → 38.900 v14.1.0 | V metadata; not read |
| R1-168013 (CR 0040) | RAN1#86 | ZTE et al. | SC across floors | Agreed | V metadata; not read |
| R1-167482 (CR 0018) | RAN1#86 | Ericsson | Correction for ZOA scaling factor | Noted (not adopted) | V metadata; not read |
| R1-167233 (CR 0006) | RAN1#86 | Huawei | Translation equation for scaling of angles | Noted | V metadata; not read |
| R1-168008/168023 (CR 0004) | RAN1#86 | Huawei | Clarifications and corrections to 7.5 fast fading | Agreed | V metadata; not read |
| R1-1610313 (CR 0051) → R1-1612895/1613063 (CR 0056) → R1-1701417 (CR 0074) | RAN1#86bis → #87 → NR-AH#1 (Jan 2017) | Ericsson (+ Fraunhofer from #87) | "Correction for correlation modelling for MF simulations" | Postponed twice, agreed at NR-AH#1. Into 38.901 v0.0.1. Probably the "same x for DS(f)" bullet and per-frequency Z_n. | V metadata; I content |
| R1-1701475/1701507 (CR 0076) → R1-1702698/1703906/**1703647** (CR 0078) | NR-AH#1 → RAN1#88 (Feb 2017) | Ericsson, Fraunhofer HHI | "Correction for channel generation for MF simulations" | Endorsed at #88. R1-1703647 is listed in the 38.901 v0.1.0 change history. 7.6.5.1 appears exactly between 38.900 v14.2.0 and 38.901 v14.0.0. | V metadata + TR diff; I that this doc holds 7.6.5.1 |
| R1-1612896/1613064/1613525/1613497 (CR 0057) | RAN1#87 (Nov 2016) | Ericsson, Fraunhofer | "Correction for spatially-consistent UT mobility modelling" | Postponed | V metadata; not read |
| R1-1701415 (CR 0072) → R1-1702704/**1703554** (CR 0083) | NR-AH#1 → RAN1#88 | Ericsson, Nokia (…) | Same title | Endorsed at #88. Procedure B text changed between 38.900 v14.2.0 and 38.901 v14.0.0: it now uses Steps 1–12 per position + 7.6.3.1, has an LOS rule, LOS √(1+K) spread substitution, and the "largest value among all frequencies" MF rule. | V metadata + TR diff; I attribution |
| R1-1612897 (CR 0058); R1-1701414 (CR 0071) | RAN1#87; NR-AH#1 | Ericsson | Correction for SC procedure | Agreed | V metadata; not read |
| R1-1701413 (CR 0070) | NR-AH#1 | Fraunhofer HHI | Normalise cluster powers | Agreed | V metadata; not read |
| R1-1701231 (CR 0077) | NR-AH#1 | NTT DOCOMO | Calibration assumptions for SC | Agreed (→ Table 7.8-5) | V metadata; not read |
| R1-1700990 | 2017 | (calibration summary) | Calibration results of additional features incl. SC (based on 38.900 v14.0.0) | Cited in 7.8.3. Metrics are cross-correlation of 3rd-cluster delay/AOA/LOS state vs UE distance and varying rates; no spread-fidelity metric. | V (TR text); doc not read |
| R1-1705802 / R1-1708817 (38.901 CR 0002) | RAN1#88bis / #89 | Ericsson, Fraunhofer | Correction for mobility model A | Postponed, then noted (not adopted) | V metadata; not read |
| R1-1705804 (CR 0004) | RAN1#88bis | Ericsson, Fraunhofer | Correction for scaling factor for ZOD generation | Agreed → v14.1.0 | V metadata; not read |
| R1-1706374/1706382 (CR 0011) | RAN1#88bis | NTT DOCOMO | Clarification for SC procedure | Agreed → v14.1.0 | V |
| R1-1709328/1709330 (CR 0015); R1-1711760 (CR 0017) | RAN1#89, NR-AH#2 (2017) | ZTE, DOCOMO; ZTE, Ericsson | Procedure A correction | Updates limited to < 1 m steps and chained from t_{k−1}. Delays are normalised by min over the *whole trajectory* (7.6-10a), avoiding per-snapshot min jumps. | V (diff v14.0.0 vs v14.3.0) |
| R1-1714545/1714631 (CR 0019) | RAN1#90 (Aug 2017) | Fraunhofer HHI | Spatially-consistent UT modelling | Agreed → v14.2.0. Probably the rotation-based Procedure A angle update (7.6-10b, R_n with fixed sign X_n). Last SSF-procedure CR. | V metadata + diff; I attribution of the specific text |
| R1-2504964 (CR 0026) and maintenance CRs 0028/0036/0040/0042/0043 | RAN1#121–#126 (2025–26) | Intel, InterDigital, Ericsson, CATT, … | Rel-19 7–24 GHz enhancements | Add 7.6.13 NF, 7.6.14 SNS, 7.6.15 cluster-number range, 7.6.16 polarisation variability. Steps 5–7 unchanged. | V (CR list, v19.5 text) |
| R1-2509551/2509552 (CR 0034/0035) | RAN1#123 | Ericsson, InterDigital | UL/DL reciprocity | UL uses the DL CIR under ideal reciprocity (BS–UT only; nothing on D2D swap) | V |
| R1-2505006 … R1-2608784 (CR 0027…0044) | RAN1#121–#126bis | Xiaomi, AT&T, OPPO, Ericsson, … | ISAC channel model (7.9) | 7.9.5.1: SC of 7.6.3 reused for UT-UT/SPST links with a new "link-correlated" type (correlated "subject to correlation distance"). No definition of distance between two links whose ends both move, and no reciprocity. Random ray coupling is fixed per drop. | V (v19.5 text) |
| TR 37.885 6.2.3 | Rel-15 V2X | — | V2X fast fading | Uses the 7.5 procedure with V2X parameters. Dual-mobility Doppler adds a moving-scatterer term (2αD/λ). The page excerpt has no SSF spatial-consistency or reciprocity rule. The same Doppler form is now in 38.901 7.6.10. | V (partial page) |

## 3. Academic and implementation sources

| Source | Year | Who | Topic | Finding | Label |
|---|---|---|---|---|---|
| Jaeckel, Raschkowski, Burkhardt, Thiele, "A Spatially Consistent Geometric D2D SSF Model for Multiple Frequencies", arXiv 1906.12149 (local tex) | 2019 | Fraunhofer HHI/IIS | SC + MF + D2D | Lists the operations that break SC: sorting (7.5-2), max-normalisation (7.5-9/-14), sign (7.5-16), random ray coupling. Says 7.6.5.1 does not ensure output = input LSPs and the TR has no D2D. Reports input-vs-output CDFs (UMi, 1/6/60 GHz): DS exact to ns, ASD/ESA within 1°, ASA limited by K, ESD about doubled. Does not compare against the 7.5 baseline output. | V |
| Jaeckel et al., "Efficient Sum-of-Sinusoids based SC for the 3GPP NR Channel Model", arXiv 1808.04659 | 2018 | HHI/IIS | SC random-field generator | About 2288 correlated variables per link. SoS with a 6-D D2D extension (TX and RX fields combined with 1/√2). Does not discuss sorting. | S (ar5iv summary) |
| Kurras, Dai, Jaeckel, Thiele, arXiv 1808.03549 | 2018 | HHI | Evaluation of 38.901 SC (QuaDRiGa 2.0) | CMD vs user distance behaves as expected. No jump or sorting analysis. | S |
| Dai, Kurras, arXiv 1903.10459 | 2019 | HHI | Massive SIMO measurements | Concludes the 3GPP SC feature does not adequately capture measured consistency; the environment dominates. | S |
| Gentile, Lai, PIMRC 2021, doi 10.1109/PIMRC50174.2021.9569532 | 2021 | NIST | Measurement check of SC-I (Proc A) | Good fit, but accuracy depends on environment | S |
| "Measurement-Based Validation of the 3GPP Spatial Consistency Procedures", IEEE doc 10378856 | ~2023 | ? | SC-A vs SC-B vs measurement | Title only seen in search results. Not read. | S (title) |
| Tataria, Haneda, Molisch, Shafi, Tufvesson, arXiv 2006.08491 (Proc. IEEE) | 2020/21 | — | History of 3GPP/ITU models | Names SC-I/SC-II; M.2412 has the same two procedures. Example of continuous Proc-A delays vs drop-based jumps. Summarises 7.6.5. | V (pdf text) |
| Ju, Rappaport, arXiv 1807.04392 (Globecom 2018) | 2018 | NYU | NYUSIM SC | Exponentially filtered SSPs plus incremental updates. No cluster birth/death, no sorting discussion, single frequency. Criticises the fixed 3GPP cluster counts. | S (ar5iv) |
| Ademaj, Schwarz, Berisha, Rupp, IEEE Access 2019, doi 10.1109/ACCESS.2019.2958154; Müller et al., Vienna 5G SLS, EURASIP JWCN 2018 | 2018/19 | TU Wien | SC model for GBSMs; Vienna SLS | SSP maps on a grid with bilinear interpolation, validated against ray tracing. No MF or D2D. | S |
| Pessoa et al., IEEE Access 2019 (UFC) | 2019 | UFC/Ericsson | SoS SC + dual mobility extension of 3GPP SCM | Correlated SSPs at both ends; some features calibrated for single mobility only. Reciprocity and sorting not known. | S (snippet) |
| Zugno et al., arXiv 1702.04822; ns-3 MR !845 (2022); CTTC 5G-LENA blog / MR !2490 (Dec 2025) | 2017–2025 | Padova, CTTC | ns-3 38.901 SC | Implements Procedure A (cluster "continuity" claimed). Validated by smoothness tests and SNR traces, not against 7.5 statistics. No MF. | S |
| Sionna PHY `tr38901` (main branch, read 2026-10-28) | 2025–26 | NVIDIA | 3GPP SLS models | Optional `enable_spatial_consistency` (default off): Cholesky-correlated delays, powers, angle randoms, XPR and phases across UTs in *one snapshot*; signs and ray coupling fixed per "track id". Keeps argsort of delays, max-normalised angles and top-2 subclusters, so it is 7.6.3.1 on top of 7.5 and inherits the jumps. Nothing retained across topology updates; single carrier. | V (source code) |
| MATLAB 5G Toolbox `nrCDLChannel` | R2023b+ | MathWorks | CDL link-level | Lists 7.6.10 dual mobility (since R2023b); 7.6.3/7.6.5 not listed (first 100 k chars read). | V (partial) |
| Zhang, Chen, Zhou, Wang et al., arXiv 2012.01728 | 2020 | SEU/PML | Multi-frequency multi-link non-stationary GBSM | SC and cross-frequency correlation at parameter initialisation, cluster birth/death. Compared with 38.901 and QuaDRiGa (CCF, PDP). | S |
| Poddar et al., arXiv 2507.19266 | 2025 | Qualcomm, Ericsson, Nokia, … | Rel-19 7–24 GHz overview | Cluster-number variability (bounded interval per scenario) and ray-number variability. SC mentioned only for NF/SNS. | S |
| Peter et al. / mmMAGIC, arXiv 1712.09435 | 2017 | mmMAGIC | DS frequency dependence | Measured DS(f) dependence, which motivates MF methods | S |

## 4. Findings per question

### Q1: 3GPP history and rationale

- **Procedures A and B** were already in TR 38.900 v14.0.0, the first approved version of 2016-06 [V via CR list].
  - The originating proposals (RAN1#84bis/#85, 2016) are not identified.
  - I could not verify who proposed Procedure B. Ericsson is the most active CR author on it later [I].
  - Rel-14 maintenance of Procedure B came from Ericsson/Nokia (CR 0072/0083, RAN1#88) [V metadata].
  - Rel-14 maintenance of Procedure A came from ZTE/DOCOMO (CR 0015/0017) and HHI (CR 0019) [V].
- **Rationale for Procedure B.** The TR text says Procedure B keeps the spatial or time evolution of delays and
  angles "within reasonable limits" [V]. This implicitly admits that 7.5 + 7.6.3.1 alone does not give smooth
  evolution.
- **7.6.5.1** came from Ericsson + Fraunhofer HHI, RAN1#88, Feb 2017.
  - The 7.6.5 Note gives the rationale: the shared-delay/angle requirement conflicts with 7.5 when DS or AS depends on
    frequency, and 7.6.5.1 is the alternative that generates delays and angles independently of frequency [V].
  - The new method draws delays and angles once from the anchor-frequency spreads with r = 1.5, then shapes the
    per-frequency powers [V].
- **Combining SC and MF.** The only explicit rule in the TR is the Procedure B "largest value among all simulated
  frequencies" sentence, present since v14.0.0 [V].
  - No Rel-15 to Rel-19 CR touched 7.5 steps 5–7, 7.6.3.2 or 7.6.5/7.6.5.1 [V, CR list].
  - I found no later RAN1 contribution on combining SC with MF or on sorting discontinuities [S, negative search].
- **D2D and sidelink.**
  - TR 37.885 reuses 7.5 and adds dual-mobility Doppler with moving scatterers. The same model is now 7.6.10
    (Doppler only) [V].
  - Rel-19 ISAC 7.9.5.1 extends SC to UT-UT links with a "link-correlated" type. It does not say how to correlate
    over the 4-D (TX, RX) position space, or what happens when TX and RX are swapped [V]. Monostatic and bistatic
    background channels are explicitly *not* spatially consistent [V].
- **Cluster-number variability.** 7.6.15 (Rel-19) lets N vary per link within [D1, D2], with the "selection determined
  by the user" [V]. Combined with the C_φ/C_θ table gaps, this re-opens the scaling-factor problem (see Q4).

**Library implication:**
- A TR-baseline path is fully specified only for the drop-based case.
- The only TR-sanctioned way to get SC + MF together is Procedure B plus the "max spread" rule. Offering a
  `38.901_B` path with MF support is therefore standards-traceable.
- D2D and reciprocity have no TR basis, so the library has to define them itself, following the Jaeckel 2019
  approach.

### Q2: Academic and simulator handling

- No open implementation I found resolves the sorting and max-normalisation discontinuities while staying 7.5
  compliant.
  - Sionna correlates the random variables but keeps sorting [V].
  - ns-3 uses Procedure A, which tracks clusters kinematically from one 7.5 draw and so avoids re-sorting [S].
  - NYUSIM and Vienna use their own SSP evolution [S].
- Multi-frequency SC is handled only by QuaDRiGa/Jaeckel 2019 [V] and by the SEU multi-frequency GBSM [S].
- Dual-mobility SC: SoS 6-D fields (Jaeckel 2018 [S]) and Pessoa 2019 [S]. Mirrored or symmetric fields for TX/RX
  reciprocity appear only in Jaeckel 2019 [V].

**Library implication:** Procedure A is the common "consistent baseline" in simulators (ns-3), but it is
trajectory-based and not a random field. A field-based path (paper method) is more general and needs its own
validation.

### Q3: Statistics comparisons

- Jaeckel 2019 compares input LSP CDFs with output spreads for its own method only [V].
- Calibration of the 7.5 baseline (7.8.2) checks coupling loss, geometry and spreads; SC calibration (7.8.3, Table
  7.8-5) checks only correlations and varying rates of the "third cluster" [V].
  - The "third cluster" is identified by sort index, so this metric itself depends on sorting [I].
- I found no study of Procedure B or 7.6.5.1 output DS/AS against input LSPs [negative search].
  - [I] Procedure B's uniform-on-[0, 2·10^(μ+σ)] delays with exp(−τ/DS) powers give an output DS that depends on how
    10^(μ+σ) compares with DS. For DS ≪ range the PDP is clipped by the power law; for DS near the range it is close
    to uniform. Some bias against the per-link DS is expected, but this is not verified.
  - [I] 7.6.5.1 likewise gives no guarantee. The g-coefficients target the per-frequency spreads, but the LOS √(1+K)
    rules and the wrapped angles bias the output.
- **Library implication:** add a validation harness that compares output DS/AS against input LSPs (and against
  baseline 7.5 output) for all paths: 38.901, 38.901_B, 38.901_MF and quadriga. This would be new evidence.

### Q4: Known errata and criticisms of steps 5–7

- **Angular-spread limits.**
  - The TR caps the log-AS: ASA/ASD at 104°, ZSA/ZSD at 52° [V per paper; TR 7.5 step 4].
  - Jaeckel 2019 shows that K further limits the achievable AS: at K = 10 dB the maximum azimuth spread is 57°, and
    in UMi LOS the TR's ASA of about 50° is not achievable with K of about 9 dB [V].
- **Wrapped Gaussian/Laplacian.** Wrapping limits the AS and gives large angle offsets for weak clusters [V, paper].
- **Rel-14 CRs on scaling factors:**
  - Ericsson ZOA scaling factor (R1-167482): noted.
  - Huawei angle-translation equation (R1-167233): noted.
  - Huawei "correction of angular scaling parameters" for CDL (R1-167238): not pursued.
  - Ericsson/HHI ZOD scaling factor (R1-1705804): agreed.
  - HHI cluster power normalisation (R1-1701413): agreed.
  - All are V metadata only; their content was not read.
- **C_φ(N), C_θ(N) tables.** I found no published derivation; they are presumably fitted. Gaps for some N and the
  non-monotonic C_θ are already noted in ssf_plan.md A8 [V, local]. With 7.6.15 a user may pick an N without a
  table value [V].
- **NYUSIM** criticises the fixed N and M as inconsistent with mmWave measurements; Rel-19 7.6.15 partly answers
  this [S].
- **Library implication:**
  - The baseline path should reproduce the TR literally, including its caps.
  - For N missing from the tables, document the interpolation or refit and flag it as non-TR.
  - Expect ASA in LOS to stay below the input in every method.

## 5. Open points and items not verified

- **3GPP Tdoc contents not read** (zip files): R1-1703647/1703906/1702698, R1-1701475/1701507, R1-1703554/1702704,
  R1-1714631, R1-1700990, R1-165974/165975, and the RAN1#84bis/#85 WFs that introduced SC, Procedure A/B and 7.6.5
  in 38.900 v14.0.0.
- Who first proposed Procedure B and the anchor-frequency method, and their stated motivations beyond the TR Note,
  are therefore unverified.
- I could not find RAN1 chairman notes on these topics in HTML. Agreements may contain further rationale.
- IEEE doc 10378856 (measurement validation of SC-A/SC-B) and "Flexible 3GPP MIMO Channel Modeling and Calibration
  With Spatial Consistency" (IEEE doc 10214216) were not readable (paywall or bot check).
- TR 37.885: I read only the 6.2.3 excerpt. Whether 37.885 references 38.901 SC for V2V SSF is unconfirmed.
- MATLAB: I did not check for a separate 38.901 system-level object, and the nrCDLChannel page was read only
  partially.
- ITU-R M.2412 SC text was not read directly; "same as 38.901" is [S].
- I found nothing on SC, MF or SSF from the Rel-20 6G study (as of 2026-10).
- I found no published quantitative comparison of baseline vs Procedure B vs 7.6.5.1 vs the paper method. This is a
  gap that the library's validation can fill.
