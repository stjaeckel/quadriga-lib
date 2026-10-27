# 3GPP channel-model calibration: references and a test set

Status of the 3GPP TR 38.901 calibration campaigns, where their results are published, which reference numbers
they give, and how to use them to calibrate the large-scale model of quadriga-lib (`include/quadriga_gbsm.hpp`)
inside QuaDRiGa-NG. Research date: 2026-10-26, target model TR 38.901 v19.5.0.

Labels:
- **[V]** verified: read in the TR, in a 3GPP file on the public FTP server, or in a local file (source named).
- **[S]** secondary source only (arXiv paper or blog), not checked against the primary document.
- **[I]** inference or recommendation.

## 1. Status

- There is no open calibration campaign as of v19.5.0 (Sept 2026) [V]. Every campaign of TR 38.901 is closed and
  cited in the TR by a fixed document number.
- The two most recent campaigns, Rel-19 7–24 GHz and Rel-19 ISAC, ran from May to Nov 2025, both against
  **TR 38.901 v19.0.0** [V]. The RAN1#122 moderator stated that no further round would follow and that the
  submitted calibrations state v19.0.0 as their basis [V, R1-2506405 §3.8].
- The corrections of v19.1 to v19.5 were not re-calibrated [V, TR Annex B]: CR 0028, 0036, 0040, 0042 (7–24 GHz
  corrections), CR 0029, 0031, 0033, 0038, 0039 (ISAC), CR 0032 (handheld UT polarized antenna), CR 0035 (UL/DL
  reciprocity), CR 0043 "Correction on ZOD generation" (RP-261752, RAN#113, 2026-09, which produced v19.5.0).
- Rel-20: no RAN1 channel-model calibration campaign was found. Out-of-scope FR3 proposals were deferred to the 6G
  study item [V, R1-2506405 §3.7]; Rel-20 sensing work (TR 38.765) shows no calibration agenda [S].

## 2. Campaigns and downloads

All documents are zip files (Word document plus Excel workbooks) under
`https://www.3gpp.org/ftp/tsg_ran/WG1_RL1/` [V]. Plain `curl` gets a bot-check page; a browser gets HTTP 200. The
Tdoc portal pages need a login. Files above about 10 MB were confirmed to exist but not read.

| Campaign | TR clause | Meeting, date | Document, FTP path | Companies | Model version | Calibrated |
|---|---|---|---|---|---|---|
| Rel-14 large-scale | 7.8.1, Table 7.8-1 | RAN1#85 Nanjing, 23–27 May 2016 | R1-165974, `TSGR1_85/Docs/R1-165974.zip` (~5 MB, read) | 19 per cover text; workbook columns 21 (UMa, UMi), 20 (InH) [V] | TR 38.900 v14.0.0 [V] | UMa, UMi-SC, InH open office at 6/30/70 GHz; coupling loss, geometry; no fast fading |
| Rel-14 full | 7.8.2, Table 7.8-2 | RAN1#85 (from the folder) | R1-165975, `TSGR1_85/Docs/R1-165975.zip` (>10 MB, not read) | not verified | TR 38.900 v14.0.0 [V] | UMa, UMi, InH at 6/30/60/70 GHz, configs 1 and 2; coupling loss, wideband SIR, serving-link DS/ASD/ZSD/ASA/ZSA, singular values |
| Rel-14 additional features | 7.8.3, Tables 7.8-3 to 7.8-6 | RAN1 NR Ad-Hoc #1 Spokane, 16–20 Jan 2017 | R1-1700990, `TSGR1_AH/NR_AH_1701/Docs/R1-1700990.zip` (read) | blockage A 10, blockage B 1, large BW 9, O2 9, spatial consistency 11 [V] | TR 38.900 v14.x | O2 absorption 60 GHz, large BW/array 30 GHz, spatial consistency 30 GHz, blockage A/B |
| Rel-16 InF | 7.8.4, Table 7.8-7 | RAN1#98, Aug 2019 (cover page says #97, a template leftover) | R1-1909704, `TSGR1_98/Docs/R1-1909704.zip` (3 MB, read) | 4: Ericsson, Nokia, ZTE, Huawei [V] | Rel-16 draft | InF-SL/DL/SH/DH at 3.5/28 GHz; coupling loss, geometry, DS/ASD/ZSD/ASA/ZSA; no first-path delay (absolute delay model not yet agreed) |
| Rel-19 7–24 GHz | 7.8.1 to 7.8.3 (Tables 7.8-1A, 7.8-2A, 7.8-6A/B/C) | RAN1#122 Bengaluru, 25–29 Aug 2025 | **R1-2506406**, `TSGR1_122/Docs/R1-2506406.zip` (>10 MB, not read); first results R1-2504791 (RAN1#121, superseded) | about 16 to 18 in the drafts [V] | **TR 38.901 v19.0.0** [V] | UMa/UMi 6 GHz rerun with v19 modelling, UMa/UMi/SMa 7 GHz (LS and full, config 3B), near field, SNS-BS, SNS-UT, UT antenna |
| Rel-19 ISAC | 7.9.6 | RAN1#123, Nov 2025 (location not verified) | **R1-2509126**, `TSGR1_123/Docs/R1-2509126.zip` (>10 MB, not read); preceded by R1-2506616 (RAN1#122) | 20+ (file names of the drafts) [V] | v19.0.0 (the TR text says "TR 38.900 V19.0.0", a typo) | UAV, human, automotive, AGV targets; target and background coupling loss, DS/AS, spatial consistency, type-2 EO |

Public draft workbooks (also on the FTP server, below the fetch limit) [V]:
- 7–24 GHz: `TSGR1_121/Inbox/drafts/9.8(FS_NR_7_24GHz_CHmod)/calibration/` with `Calibration_LS_v03..v31`,
  `Calibration_Full_v03..v34`, `Calibration_NF_v04..v31`, `Calibration_SNS_BS_v02..v16`,
  `Calibration_SNS_UE_v02..v14`, `Calibration_UTant_v00..v04`. The last LS version,
  `Calibration_LS_v31_Apple_Ericsson.xlsx` (19 Aug 2025), includes the RAN1#122 updates; it is very likely the
  workbook of R1-2506406 [I]. Contributors: Huawei, InterDigital, Samsung, ZTE, CATT, Lenovo, Apple, Qualcomm,
  Nokia, Ericsson, OPPO, MediaTek, Intel, Sharp, Spreadtrum, ETRI, xiaomi, vivo (SMa only).
- ISAC: `TSGR1_121/Inbox/drafts/9.7(FS_Sensing_NR)/[Post-121-ISAC-02] Calibrations/`.

Workbook layout [V]: one sheet per scenario and frequency (e.g. `UMi-6GHz`, `SMa-7GHz-1299m`); row 25 the
companies, row 28 the metric titles, rows 29 to 129 the percentiles 0 to 100 with one column per company plus a
"Mean" column. The InF workbook: row 1 titles, row 2 companies, rows 3 to 103 percentiles.

Points to know [V unless marked]:
- InF hall height: the e-mail discussion proposed 25 m (sub-scenarios 1, 3) and 15 m (2, 4); Table 7.8-7 says
  10 m. Which value the companies used is not known.
- InF cluster scaling: ZTE used C_φ(25) = 1.289 and C_θ(25) = 1.178, the others the N = 20 factors.
- The Rel-14 results used 38.900 v14.0.0; UMa-LOS path loss, the InH DS frequency dependence and the InH ASA changed
  later, so the 2016 curves are outdated for UMa and InH.
- The Rel-19 6 GHz rerun changes the reference substantially: median serving-link ASD UMa 6 GHz (config 1) 8.3°
  against 20.2° in 2016; median DS UMi 155.7 ns against 196.6 ns.

## 3. Metrics and the library functions they exercise

| Library function | Exercised by | Notes |
|---|---|---|
| `path_loss` (all scenarios, frequency terms) | Coupling-loss CDF of the large-scale campaigns at each frequency | The shift between the 6/30/70 GHz curves isolates the frequency terms |
| `los_probability` | Shape (bimodality) of the coupling-loss CDF, geometry; InF clutter parameters | Indirect only, no LOS-fraction metric is reported [I] |
| `o2i_loss` | UMa/UMi large-scale (50 % low / 50 % high loss); SMa 7 GHz (low-loss A, car window, 90 % residential / 10 % commercial) | SMa is the only campaign that checks the Rel-19 O2I variants |
| SF (`lsp_generate` with `sf_sigma` of `path_loss`) | Tails of the coupling-loss CDF, geometry spread | |
| `lsp_generate` DS, ASD, ASA, ESD (ZSD), ESA (ZSA) | Serving-link spread CDFs of the full calibrations, InF spreads | The spreads come from the generated clusters and rays and are affected by serving-cell selection and the K-factor; an LSP-only check is approximate [I]: closest for DS and ZSD, worst for ASA/ASD in LOS |
| ESD distance and height terms | ZSD CDF of the full calibration (UMa median about 1.8 to 1.9°, UMi about 0.9°) [V] | |
| `eod_offset_deg` (ZOD offset) | No direct metric [I]: spreads are invariant to a common shift; the offset enters only coupling loss and SIR of the full calibration with a down-tilted array | CR 0043 postdates all calibrations: unit tests against TR Tables 7.5-7 to 7.5-12 are the only exact check |
| `lsf_init_phases`, `lsf_draw` | Spatial-consistency cases of 7.8.3 (cross-correlation of LOS state, cluster delay/AoA at 30 GHz) | Mostly SSF variables |

## 4. Reference numbers

Format p5 / p50 / p95 (InF: p10 / p50 / p90); coupling loss and geometry in dB, DS in ns, angles in degrees;
[min..max] = range of the company values at p50.

Rel-14 large-scale, R1-165974, "Mean" column of 20 to 21 companies [V]:

| Scenario | Coupling loss | Geometry SIR (no noise) |
|---|---|---|
| UMi 6 / 30 / 70 GHz | −142.3 / −112.2 [−114.9..−110.5] / −74.8; −164.6 / −132.4 / −89.3; −184.2 / −149.1 / −96.8 | −1.7 / 7.3 / 24.9 (6 GHz) |
| UMa 6 / 30 / 70 GHz | −150.4 / −120.3 [−123.1..−118.1] / −86.0; −172.0 / −140.1 / −102.0; −191.4 / −156.6 / −110.7 | −1.0 / 8.5 / 25.5 (6 GHz) |
| InH 6 / 30 / 70 GHz | −71.1 / −59.3 [−60.0..−58.5] / −45.0; −85.0 / −73.3 / −59.1; −92.4 / −80.7 / −66.4 | −4.4 / 3.0 / 17.9 (6 GHz) |

Rel-19 large-scale, v19.0.0, draft `Calibration_LS_v31`, median of the companies [V]:

| Scenario | Coupling loss | p50 range | Geometry SIR p50 |
|---|---|---|---|
| UMi 6 GHz (15 companies) | −142.6 / −112.2 / −75.8 | [−115.9..−106.1] | 7.1 |
| UMa 6 GHz (15) | −151.2 / −120.5 / −87.2 | [−125.1..−118.5] | 8.3 |
| UMi 7 GHz (16) | −144.3 / −113.5 / −77.0 | [−118.1..−107.5] | 7.0 |
| UMa 7 GHz (16) | −153.2 / −122.0 / −88.5 | [−126.5..−119.1] | 8.1 |
| SMa 7 GHz, ISD 1299 m (15) | −123.5 / −103.4 / −85.6 | [−121.5..−96.3], Lenovo an outlier | 6.3 |
| SMa 7 GHz, ISD 1732 m (11) | −133.4 / −108.6 / −89.1 | [−129.4..−106.5] | 6.6 |

Inter-company spread at the median: about ±5 dB in Rel-19 against about ±2 dB in 2016 [V].

Rel-19 full calibration, June 2025 draft `Calibration_Full_v20`, median of the companies [V; the final R1-2506406
may differ slightly]:

| Scenario | DS | ASD | ZSD | ASA | ZSA |
|---|---|---|---|---|---|
| UMi 6 GHz, config 1 (13) | 22 / 156 / 623 | 6.0 / 19.8 / 79 | 0.3 / 0.9 / 4.7 | 24 / 57 / 92 | 3.6 / 9.7 / 38 |
| UMa 6 GHz, config 1 (13) | 42 / 189 / 748 | 2.4 / 8.3 / 42 | 0.2 / 1.8 / 12.7 | 30 / 61 / 94 | 3.7 / 11.6 / 38 |
| UMi 7 GHz, config 3B (12), p50 | 154 | 18.4 | 0.9 | 57 | 9.5 |
| UMa 7 GHz, config 3B (12), p50 | 194 | 8.3 | 1.9 | 60 | 11.3 |
| SMa 7 GHz, 1299 m (12), p50 | 65 | 3.9 | 1.5 | 25 | 14 |

Further p50 values, config 1: coupling loss UMi 6 GHz −105.8, UMa 6 GHz −118.6; wideband SIR UMi 3.7, UMa 5.8.

Rel-14 full calibration, R1-165975, as stored in QuaDRiGa-NG (v14.0.0 parameters), 6 GHz [V, local file]: UMi DS
27.7 / 196.6 / 787.5, p50 ASD 19.7, ZSD 0.87, ASA 60.3, ZSA 11.1; UMa DS 48.5 / 230.9 / 956.3, p50 ASD 20.2,
ZSD 1.93, ASA 62.8, ZSA 12.9; InH p50 DS 15.8, ASD 35.3, ZSD 11.2, ASA 46.2, ZSA 11.8.

InF, R1-1909704, median of 4 companies, p10 / p50 / p90 [V]:

| Sub-scenario | Coupling loss 3.5 GHz | 28 GHz | DS | ASD | ZSD | ASA | ZSA |
|---|---|---|---|---|---|---|---|
| SL | −66.4 / −60.5 / −51.8 | −83.7 / −77.7 / −69.1 | 94 / 174 / 299 | 11 / 29 / 68 | 6 / 16 / 38 | 11 / 33 / 78 | 6.5 / 18 / 40 |
| DL | −81.7 / −73.1 / −60.3 | −100.0 / −91.3 / −78.5 | 99 / 171 / 299 | 22 / 39 / 76 | 7 / 17 / 45 | 23 / 56 / 105 | 11 / 26 / 47 |
| SH | −74.3 / −69.0 / −62.2 | −91.7 / −86.2 / −79.4 | 120 / 222 / 393 | 11 / 28 / 65 | 6 / 16 / 38 | 10 / 29 / 73 | 6 / 17 / 38 |
| DH | −68.8 / −64.4 / −59.3 | −86.9 / −82.3 / −76.9 | 80 / 141 / 242 | 20 / 38 / 74 | 7 / 17 / 40 | 21 / 52 / 105 | 11 / 23 / 42 |

Spreads at 3.5 GHz. Geometry p50 (with noise): SL 2.1, DL 5.3, SH 0.0, DH 0.0. ASA values above 100° suggest
non-circular or other AS definitions at some companies [I].

## 5. QuaDRiGa-NG reference data is stale

`/home/stephan/sjc/QuaDRiGa-NG/tutorials/calib_3GPP_ref_data.m` [V]:
- `cl36873a/b`, `gf36873a`, `zod36873a`, `sinr36873b`, `zsb/zsa36873b`, `sv*36873b`: TR 36.873 phases 1 and 2
  (2 GHz, not a 38.901 calibration).
- `cl38900a`, `gf38900a`: R1-165974 (the 5:5:95 percentiles of the Mean column; geometry without noise).
- `cl38900b`, `sinr38900`, `ds/asd/zsd/asa/zsa38900`, `sv1/sv2/svR_38900`: R1-165975.
- All are Mean-column percentiles of 2016, based on 38.900 v14.0.0. There is no InF, Rel-19 (6/7 GHz with v19,
  SMa) or ISAC data.

`calib_3GPP_38901_large_scale.m` and `calib_3GPP_38901_full.m` [V] cite TR 38.901 V14.1.0 and note mismatches for
UMa and InH caused by the 38.900 to 38.901 changes. The large-scale script drops 2000 UEs (19 sites × 3 sectors,
ISD 200/500/20 m, 80 % indoor, floors as in TR 36.873), which is small for a 5 % tail [I].

## 6. Recommended calibration test set [I]

Run each case as a system-level drop through the QuaDRiGa-NG calibration tutorials with the large-scale functions
of the library, at least 20,000 serving links per case. Compare p5 / p50 / p95 with the median of the companies.
Acceptance: inside the company min..max band at every percentile, and within 1 dB (coupling loss, geometry) or
10 % (spreads) of the company median at p50.

| Case | Layout | Metric | Reference | Exercises |
|---|---|---|---|---|
| UMa, UMi 6 GHz large-scale | Table 7.8-1: 19×3 sites, ISD 500/200 m, 80 % indoor, 50/50 O2I | coupling loss, SIR without noise | R1-2506406 (draft `Calibration_LS_v31`, sheets UMi-6GHz, UMa-6GHz); R1-165974 only to show the drift since v14 | path loss, LOS probability, O2I, SF |
| UMa, UMi 7 GHz large-scale | Tables 7.8-1, 7.8-1A | same | `Calibration_LS_v31` 7 GHz sheets | frequency dependence in FR3 |
| SMa 7 GHz, ISD 1299 and 1732 m | Table 7.8-1A (tilt 95°/92°, 49 dBm, residential/commercial, car, low-loss A) | same | `Calibration_LS_v31` SMa sheets (median; outliers) | SMa path loss and LOS probability, O2I low-loss A, car penetration |
| InH open office 6/30/70 GHz | Table 7.8-1 | coupling loss, geometry | R1-165974 (only source; check the InH changes since v14 first) | InH path loss and LOS probability |
| InF-SL/DL/SH/DH 3.5/28 GHz | Table 7.8-7 | coupling loss, geometry, optionally DS/AS | R1-1909704 (4 companies; hall-height ambiguity) | InF path loss, clutter LOS probability, InF LSPs |
| UMa, UMi 6 GHz full, config 1 | Table 7.8-2 with 7.8-2A | serving-link DS/ASD/ZSD/ASA/ZSA, coupling loss, SIR | `Calibration_Full` (final R1-2506406) | LSP generation, cross-correlation, ZSD law; ZOD offset indirectly |
| UMa, UMi, SMa 7 GHz full, config 3B | Table 7.8-2A | same | same | Rel-19 LSP tables at 7 GHz |
| Unit tests (not 3GPP) | fixed (d2D, hUT, fc) | path loss, p_LOS, O2I, LSP mean/std, ZOD offset | TR v19.5.0 formulas | exact; the only check of CR 0043 and the other post-v19.0.0 corrections |

Further recommendations [I]:
- Use R1-2506406 (v19.0.0) as the primary reference for UMa/UMi and add SMa at 7 GHz; download the zip in a browser
  and diff it against the drafts `Calibration_LS_v31` and `Calibration_Full_v34`.
- Extend `calib_3GPP_ref_data.m` with the Rel-19 6/7 GHz and SMa curves and the InF curves, store the company
  min..max band besides the median (company spreads ±2 to 5 dB), and raise the UE count well above 2000.
- A residual difference in UMa NLOS coupling loss or SIR may come from the corrections after v19.0.0 (e.g. CR 0043),
  which no campaign covers.

## 7. Not verified

- The contents of R1-165975, R1-2504791, R1-2506406, R1-2506616 and R1-2509126 (above the fetch limit; only their
  existence was confirmed). The Rel-19 numbers above come from the public draft workbooks.
- The number of companies in R1-165975 and the final counts in R1-2506406 and R1-2509126.
- Location and dates of RAN1#123; the date of R1-165975 (assumed RAN1#85 from its folder).
- Whether any Rel-20 work (6G study, FS_Sensing_NR_Ph2) defines a new calibration.
- The InF hall height the companies actually used (25/15 m in the e-mail discussion, 10 m in Table 7.8-7).
- Whether the Rel-19 changes of InH or InF parameters invalidate the 2016 InH and 2019 InF references.
- Secondary sources: https://www.6ghow.com/7-24GHz_channel_model/, arXiv 2602.07623 (Rel-19 FR3 tutorial),
  2606.07328 (ISAC simulator calibration against R1-2506616), 2507.19266 (Rel-19 overview), 2608.15283,
  2610.05694; not checked against primary documents except where noted.

## Appendix: detailed source notes

Verbatim from the original research (2026-10-26). Labels: [V] verified in the primary document, [S] secondary
source, [I] inference.

### Campaign table with download paths and participants

All links point under `https://www.3gpp.org/ftp/tsg_ran/WG1_RL1/`.

| # | Campaign | TR clause | Meeting / date | Result document | Companies | Model version used | Public? |
|---|---|---|---|---|---|---|---|
| 1 | Large-scale calibration, Rel-14 ("Phase 1"): UMa, UMi-SC, InH-open office at 6/30/70 GHz | 7.8.1, Table 7.8-1 | RAN1#85 Nanjing, 23–27 May 2016 (e-mail [85-19], Samsung) | **R1-165974** `TSGR1_85/Docs/R1-165974.zip` (docx + `Phase1Calibration_v42_CMCC.xlsx`, ~5 MB) | Cover text says 19 [V]. The workbook has 21 company columns for UMa/UMi and 20 for InH [V]. | TR 38.900 v14.0.0 [V, TR] | Yes [V, downloaded] |
| 2 | Full calibration, Rel-14 ("Phase 2"): UMa, UMi-SC, InH at 6/30/60/70 GHz, configs 1 and 2 | 7.8.2, Table 7.8-2 | RAN1#85 (number range) | **R1-165975** `TSGR1_85/Docs/R1-165975.zip` (>10 MB) | not verified | TR 38.900 v14.0.0 [V, TR] | File exists (HTTP 200, >10 MB) [V]; content not read |
| 3 | Additional features, Rel-14 ("Phase 3"): O2 absorption at 60 GHz, large BW/array at 30 GHz, spatial consistency at 30 GHz, blockage A/B at 30 GHz | 7.8.3, Tables 7.8-3..6 | RAN1 NR Ad-Hoc #1 Spokane, 16–20 Jan 2017 (e-mail [87-35], Samsung) | **R1-1700990** `TSGR1_AH/NR_AH_1701/Docs/R1-1700990.zip` (4 xlsx) | Blockage A: 10; blockage B: 1; large BW: 9; O2: 9; spatial consistency: 11 [V] | TR 38.900 v14.x | Yes [V] |
| 4 | InF large-scale, Rel-16: InF-SL/DL/SH/DH at 3.5/28 GHz | 7.8.4, Table 7.8-7 | Document is in the `TSGR1_98` folder (RAN1#98, Aug 2019); its cover page says "#97 Reno, May 2019" (template leftover) [V] | **R1-1909704** `TSGR1_98/Docs/R1-1909704.zip` (docx + xlsx, 3 MB) | **4**: Ericsson, Nokia, ZTE, Huawei [V] | Rel-16 draft (v16.0.0-era) | Yes [V] |
| 5a | Rel-19 7–24 GHz, first results: LS, full, NF, SNS-BS, SNS-UT, UT antenna | 7.8.1–7.8.3 (Tables 7.8-1A, 7.8-2A, 7.8-6A/B/C) | RAN1#121 (May 2025) | **R1-2504791** `TSGR1_121/Docs/R1-2504791.zip` (>10 MB) | – | v18 → v19 drafts | File exists [V]; superseded |
| 5b | Rel-19 7–24 GHz, final results | same | RAN1#122 Bengaluru, 25–29 Aug 2025 (moderator InterDigital) | **R1-2506406** `TSGR1_122/Docs/R1-2506406.zip` (>10 MB) | Draft workbooks show about 16–18 companies (list below) [V] | **TR 38.901 v19.0.0** [V, R1-2506405] | File exists [V]; content not read (size). Draft workbooks are public [V] |
| 6a | Rel-19 ISAC: LS and full for UAV/human/automotive/AGV, plus spatial consistency and type-2 EO | 7.9.6 | RAN1#122 (moderator AT&T); draft R1-2506225, final **R1-2506616** [S: arXiv 2606.07328] | `TSGR1_122/Docs/R1-2506616.zip` (>10 MB) | 20+ (file-name chain, list below) [V] | v19.0.0 | File exists [V] |
| 6b | Rel-19 ISAC, re-calibration | 7.9.6 (the TR cites this one) | RAN1#123 (Nov 2025; location not verified). Drafts folder `RAN1_123_rev` dated 2025-11-03 | **R1-2509126** `TSGR1_123/Docs/R1-2509126.zip` (>10 MB) | – | v19.0.0 (the TR text says "TR 38.900 V19.0.0", a typo in 7.9.6.2/3) | File exists [V] |
| (ref) | 3D-channel model calibration, phases 1 and 2 (36.873, 2 GHz) | – | Rel-12 | (QuaDRiGa `cl36873a/b`, `zod36873a` …) | – | TR 36.873 | Older and not a 38.901 calibration |

**Rel-19 7–24 GHz contributors** [V, from draft workbooks]: Huawei, InterDigital, Samsung, ZTE,
CATT, Lenovo, Apple, Qualcomm, Nokia, Ericsson, OPPO, MediaTek, Intel, Sharp, Spreadtrum, ETRI,
xiaomi, and vivo (SMa only). The drafts folder is
`TSGR1_121/Inbox/drafts/9.8(FS_NR_7_24GHz_CHmod)/calibration/` and contains:

- `Calibration_LS_v03..v31`
- `Calibration_Full_v03..v34`
- `Calibration_NF_v04..v31`
- `Calibration_SNS_BS_v02..v16`
- `Calibration_SNS_UE_v02..v14`
- `Calibration_UTant_v00..v04`

The last LS version, `Calibration_LS_v31_Apple_Ericsson.xlsx` (19 Aug 2025), says it includes the
RAN1#122 updates by Huawei (R1-2505225), ZTE (R1-2505499), Samsung, Sharp (R1-2505653), Intel
(R1-2505766), Apple (R1-2505884) and Ericsson (R1-2506374). It is very likely, though not checked,
the workbook that went into R1-2506406 [I].

**ISAC contributors** [V, from file names in `[Post-121-ISAC-02] Calibrations/UAV-UMa-AV/`]:
Tiami, DOCOMO, LGE, Huawei, Lenovo, CTC, Ericsson, xiaomi, vivo, OPPO, CMCC, NIST, ZTE, CATT,
BUPT, InterDigital, ITRI, Qualcomm, Nokia, Samsung, Spreadtrum, Sony and IDC. The folder is under
`TSGR1_121/Inbox/drafts/9.7(FS_Sensing_NR)/`. The cover page draft says "At least xx companies".

Access notes:

- Plain `curl` on the 3GPP FTP returns a bot-check page. A browser or WebFetch gets HTTP 200.
- The portal Tdoc pages (`portal.3gpp.org/ngppapp/CreateTdoc.aspx`) need a login.
- Some draft subfolder paths returned HTTP 403.


### Full source list

- TR 38.901 v19.5.0, clauses 7.8 and 7.9.6 and Annex B (local copy).
- 3GPP FTP, public, under `https://www.3gpp.org/ftp/tsg_ran/WG1_RL1/`:
  - `TSGR1_85/Docs/R1-165974.zip`, `TSGR1_85/Docs/R1-165975.zip`
  - `TSGR1_AH/NR_AH_1701/Docs/R1-1700990.zip`
  - `TSGR1_98/Docs/R1-1909704.zip`
  - `TSGR1_121/Docs/R1-2504791.zip`
  - `TSGR1_122/Docs/R1-2506405.zip` (FL summary #2, 7–24 GHz maintenance), `R1-2506406.zip`,
    `R1-2506616.zip`
  - `TSGR1_123/Docs/R1-2509126.zip`
  - `TSGR1_121/Inbox/drafts/9.8(FS_NR_7_24GHz_CHmod)/calibration/`
  - `TSGR1_121/Inbox/drafts/9.7(FS_Sensing_NR)/[Post-121-ISAC-02] Calibrations/`
- Local QuaDRiGa-NG files: `/home/stephan/sjc/QuaDRiGa-NG/tutorials/calib_3GPP_ref_data.m`,
  `calib_3GPP_38901_large_scale.m` and `calib_3GPP_38901_full.m`.
- Secondary:
  - https://www.6ghow.com/7-24GHz_channel_model/
  - https://arxiv.org/html/2606.07328 (ISAC simulator calibration against R1-2506616)
  - https://arxiv.org/html/2602.07623 (Rel-19 FR3 tutorial)
  - https://arxiv.org/abs/2507.19266 (Rel-19 overview)
