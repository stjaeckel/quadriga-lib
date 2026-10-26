# 3GPP channel-model calibration: status, sources, and a test set for the large-scale port

Research date: 2026-10-26. Target model: TR 38.901 v19.5.0 (local copy
`/home/stephan/sjc/references/3gpp_38901/38901-v19.5.0.md`).

Labels used in this report:

- **[V]** verified. The fact was read in the TR, in a 3GPP file on the public FTP server, or in a
  local file. The source is named.
- **[S]** secondary source only (arXiv paper or blog).
- **[I]** my inference or recommendation.

Downloads: the WebFetch tool saved several public 3GPP files and their spreadsheets locally as a
side effect (session tool-results, with copies under the session scratchpad). These were
R1-165974, R1-1700990, R1-1909704, R1-2506405, the ISAC cover-page draft, two Rel-19 calibration
workbooks from the RAN1#121 drafts folder, and the arXiv 2507.19266 PDF. The numbers below were
extracted from them with small read-only Python scripts. Nothing in the repository was changed
except this report.

---

## 1. Is calibration still ongoing? Which campaigns exist?

Short answer: there is **no open calibration campaign** as of v19.5.0 (Sept 2026).

- Every campaign in TR 38.901 is closed and cited by a fixed document number.
- The two most recent campaigns, Rel-19 7–24 GHz and Rel-19 ISAC, ran from May to Nov 2025. Both
  were done against **TR 38.901 v19.0.0**.
- The RAN1#122 moderator said explicitly that there would be no further rounds [V, R1-2506405
  §3.8]. The quote: "even if we update the model, moderator thinks we should not perform another
  round of calibrations. The submitted calibrations will clear state that calibration was
  performed based on v19.0.0."
- After v19.0.0, the TR changed in v19.1–v19.5 through these CRs [V, TR Annex B]:
  - CR 0028, 0036, 0040 and 0042: 7–24 GHz corrections.
  - CR 0029, 0031, 0033, 0038 and 0039: ISAC.
  - CR 0032: handheld UT polarized antenna model.
  - CR 0035: UL/DL reciprocity.
  - CR 0043, "Correction on ZOD generation" (RP-261752, RAN#113, 2026-09), which produced
    v19.5.0.
- None of these changes was re-calibrated.
- Rel-20: I found **no** Rel-20 RAN1 channel-model calibration campaign.
  - At RAN1#122, out-of-scope FR3 channel-model proposals (SKT) were deferred to the 6G study
    item, agenda 11.x [V, R1-2506405 §3.7].
  - Rel-20 sensing work (FS_Sensing_NR_Ph2, TR 38.765) exists, but I saw no calibration agenda in
    it [S].
  - This is marked as unverified in §6.

### Campaign table

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

## 2. Where are the results published?

- The TR contains only the conditions and metrics (7.8, 7.9.6) and a pointer to one Tdoc per
  campaign [V]. The results are zipped Word documents with Excel attachments on the public FTP
  server under `ftp/tsg_ran/WG1_RL1/TSGR1_<mtg>/Docs/` [V]:

  | Campaign | Document | Folder |
  |---|---|---|
  | Rel-14 large-scale | R1-165974 | TSGR1_85 |
  | Rel-14 full | R1-165975 | TSGR1_85 |
  | Rel-14 additional features | R1-1700990 | TSGR1_AH/NR_AH_1701 |
  | InF | R1-1909704 | TSGR1_98 |
  | Rel-19 7–24 GHz | R1-2506406 (superseding R1-2504791) | TSGR1_122 |
  | ISAC | R1-2509126 (preceded by R1-2506616) | TSGR1_123 |

- Spreadsheet format [V, all workbooks read]:
  - One sheet per scenario and frequency (for example `UMi-6GHz`, `SMa-7GHz-1299m`,
    `UMa-7GHz-Config3B-20MHz`).
  - Row 25 lists the companies. Row 28 holds the metric titles. Rows 29–129 hold the percentiles
    0..100 in steps of 1, with one column per company plus a "Mean" column.
  - The InF workbook uses row 1 for titles, row 2 for companies, and rows 3–103 for percentiles
    0..100.
- The draft folders under `TSGR1_1xx/Inbox/drafts/...` are also public. They hold the
  per-revision workbooks, including files under the 10 MB fetch limit.

## 3. What is calibrated in each campaign

| Campaign | Scenarios / frequencies | Fast fading | Metrics |
|---|---|---|---|
| 7.8.1 Large-scale (R1-165974) | UMa, UMi-SC, InH open office; 6/30/70 GHz. BS (M,N,P)=(10,1,1) with 102° tilt (110° for InH). UT isotropic, single element. O2I 50 % low / 50 % high loss. 36.873 3D dropping. Wrap-around. Attachment by path loss "considering LOS angle". | none | 1) serving-cell coupling loss CDF; 2) geometry with and without noise |
| 7.8.1 additional, Table 7.8-1A (R1-2506406) | **UMa and UMi at 6 GHz with updated (v19) modelling** plus **UMa, UMi, SMa (ISD 1299/1732 m) at 7 GHz**. SMa: 20 % outdoor; indoor split 90 % residential / 10 % commercial; 0 % vegetation; O2I low-loss A; non-metallic car window. | none | coupling loss; geometry SINR with noise; geometry SIR without noise |
| 7.8.2 Full (R1-165975) | UMa, UMi, InH at 6/30/60/70 GHz. Config 1: 4x4x2 with 2 panels. Config 2: 2x2 elements, 1 element per port. UT X-pol. | yes | coupling loss; wideband SIR; DS/ASD/ZSD/ASA/ZSA of serving link (circular AS per 25.996); 1st and 2nd PRB singular values and their ratio |
| 7.8.2 additional, Table 7.8-2A (R1-2506406) | UMa/UMi 6 GHz with configs 1 and 2 (v19); UMa/UMi/SMa 7 GHz with config 3 (8x16x2, UT config B handheld, 4 ports); optional config 4 at 7/15 GHz with 200 MHz | yes | as above, plus 5) handheld UT field pattern |
| 7.8.3 Additional features (R1-1700990 and R1-2506406) | O2 at 60 GHz; large BW (2 GHz) and array at 30 GHz; spatial consistency at 30 GHz (cross-correlation of the 3rd cluster delay/AoA, LOS state, H; rate of change); blockage A/B. Rel-19 adds near field (UMi/InH at 7/15 GHz), SNS-BS and SNS-UT | yes | feature-specific; see the TR tables |
| 7.8.4 InF (R1-1909704) | InF-SL/DL/SH/DH; 3.5/28 GHz; 18 BS; isotropic antennas; attachment by path loss | serving-link spreads computed | coupling loss; geometry with and without noise; DS/ASD/ZSD/ASA/ZSA (Annex A.1). First-path excess delay was **not** delivered because the absolute-delay model was not yet agreed [V, TR and doc]. |
| 7.9.6 ISAC (R1-2509126) | UAV in UMa-AV; human in InH/InF-SH/UMa/UMi; automotive in urban grid/highway; AGV in InF-SH/DH; sensing modes TRP mono, TRP-TRP, TRP-UE, UE-UE, UE mono | LS and full | coupling loss for the target channel (PL1+PL2+RCS+SF1+SF2) and the background channel; DS/AS for full; spatial-consistency correlations; type-2 EO ray CDFs |

Points to know about these campaigns:

- **InF hall height** [V, R1-1909704 §2]: the e-mail discussion proposed hall heights of 25 m
  (sub-scenarios 1/3) and 15 m (2/4). TR Table 7.8-7 says room height 10 m. The InF reference
  curves may therefore have been produced with a different height than the TR states.
- **InF cluster scaling** [V, R1-1909704 §2]: ZTE used C_φ(25)=1.289 and C_θ(25)=1.178. The
  others used the N=20 scaling factors.
- **Rel-14 model version** [V, TR, and QuaDRiGa tutorial comments]: the Rel-14 results used
  38.900 v14.0.0. UMa-LOS path loss, the InH DS frequency dependence, and the InH ASA changed
  afterwards. The 2016 curves are therefore **outdated for UMa and InH**.
- **Rel-19 UMa ASD** [V]: the Rel-19 6 GHz rerun with v19.0.0 changes the reference
  substantially. The median serving-link ASD for UMa at 6 GHz (config 1) is **8.3°**, against
  20.2° in 2016. The median DS for UMi is 155.7 ns, against 196.6 ns in 2016.

## 4. How this informs the large-scale port

### 4.1 Which metrics test which function

| Function being ported | Directly tested by | Notes |
|---|---|---|
| Path loss (LOS/NLOS, all scenarios, frequency dependence) | LS coupling-loss CDF at each frequency (7.8.1, 7.8-1A, InF) | Coupling loss = TX power-independent gain, including the BS pattern towards the LOS direction. The frequency shift between the 6/30/70 GHz curves isolates the frequency terms. |
| LOS probability | Coupling-loss CDF shape (bimodality) and geometry; InF clutter LOS (r, h_c, d_clutter) | Indirect only. No LOS-fraction metric is reported. [I] Add your own unit test against the closed-form P_LOS. |
| O2I (high/low loss, low-loss A, car penetration, indoor distance) | UMa/UMi LS (50/50 high/low) and SMa 7 GHz (low-loss A, car window, residential/commercial split) | SMa is the only campaign that checks the Rel-19 O2I variants. |
| SF | Lower and upper tails of the coupling-loss CDF; geometry spread | |
| DS/ASD/ASA/ZSD/ZSA LSPs | Full-calibration serving-link spread CDFs (7.8.2, 7.8-2A); InF spreads | Spreads come from the generated clusters and rays, not the LSP draws. The serving-cell selection bias (SF correlation) and the K-factor matter. A port that only draws LSPs can be compared only approximately [I]: closest for DS and ZSD, worst for ASA/ASD in LOS. |
| ZSD model (d2D/hUT dependence) | Full-calibration ZSD CDF for UMa/UMi | UMa ZSD median is about 1.8–1.9°; UMi is about 0.9° [V]. |
| ZOD offset μ_offset,ZOD | **No direct metric.** | The spread metrics are invariant to a common shift. In LS calibration only the LOS angle is used. ZOD offset enters only the coupling loss and SIR of the **full** calibration with the down-tilted BS array (config 1, UMa/UMi NLOS) [I]. CR 0043 ("Correction on ZOD generation", v19.5.0) is **after** all calibrations. A unit test against TR Tables 7.5-7..12 is the only exact check. 36.873 phase-1 `zod36873a` (serving-link ZoD CDF, 2 GHz) used the 36.873 offset formula, which differs from 38.901, so it is only a qualitative check. |

### 4.2 Reference values (percentiles across companies)

Format: p5 / p50 / p95 in dB or ns or degrees; [min..max] across companies at p50.

**Rel-14 large-scale (R1-165974)** [V; "Mean" column, 20–21 companies]. These are identical to
`cl38900a` and `gf38900a` in `calib_3GPP_ref_data.m`, which are the 5:5:95 percentiles of the
Mean column; the geometry there is the SIR **without** noise.

| Scenario | Coupling loss | Geometry SIR |
|---|---|---|
| UMi 6 GHz | −142.3 / −112.2 [−114.9..−110.5] / −74.8 | −1.7 / 7.3 / 24.9 |
| UMi 30 GHz | −164.6 / −132.4 / −89.3 | – |
| UMi 70 GHz | −184.2 / −149.1 / −96.8 | – |
| UMa 6 GHz | −150.4 / −120.3 [−123.1..−118.1] / −86.0 | −1.0 / 8.5 / 25.5 |
| UMa 30 GHz | −172.0 / −140.1 / −102.0 | – |
| UMa 70 GHz | −191.4 / −156.6 / −110.7 | – |
| InH 6 GHz | −71.1 / −59.3 [−60.0..−58.5] / −45.0 | −4.4 / 3.0 / 17.9 |
| InH 30 GHz | −85.0 / −73.3 / −59.1 | – |
| InH 70 GHz | −92.4 / −80.7 / −66.4 | – |

**Rel-19 large-scale, v19.0.0** [V; `Calibration_LS_v31` draft, median of companies]:

| Scenario | Coupling loss p5 / p50 / p95 | p50 range across companies | Geometry SIR (no noise) p50 |
|---|---|---|---|
| UMi 6 GHz (n=15) | −142.6 / −112.2 / −75.8 | [−115.9..−106.1] | 7.1 |
| UMa 6 GHz (n=15) | −151.2 / −120.5 / −87.2 | [−125.1..−118.5] | 8.3 |
| UMi 7 GHz (n=16) | −144.3 / −113.5 / −77.0 | [−118.1..−107.5] | 7.0 |
| UMa 7 GHz (n=16) | −153.2 / −122.0 / −88.5 | [−126.5..−119.1] | 8.1 |
| SMa 7 GHz, 1299 m (n=15) | −123.5 / −103.4 / −85.6 | [−121.5..−96.3]; Lenovo is an outlier | 6.3 |
| SMa 7 GHz, 1732 m (n=11) | −133.4 / −108.6 / −89.1 | [−129.4..−106.5] | 6.6 |

The spread across companies is wider than in 2016: about ±5 dB at the median, against about
±2 dB [V].

**Rel-19 full calibration, June 2025 draft** [V; `Calibration_Full_v20`, median of companies;
the final R1-2506406 may differ slightly]:

| Scenario | DS (ns) | ASD (°) | ZSD (°) | ASA (°) | ZSA (°) |
|---|---|---|---|---|---|
| UMi 6 GHz, config 1 (n=13) | 22 / 156 / 623 | 6.0 / 19.8 / 79 | 0.3 / 0.9 / 4.7 | 24 / 57 / 92 | 3.6 / 9.7 / 38 |
| UMa 6 GHz, config 1 (n=13) | 42 / 189 / 748 | 2.4 / 8.3 / 42 | 0.2 / 1.8 / 12.7 | 30 / 61 / 94 | 3.7 / 11.6 / 38 |
| UMi 7 GHz, config 3B (n=12) | – / 154 / – | – / 18.4 / – | – / 0.9 / – | – / 57 / – | – / 9.5 / – |
| UMa 7 GHz, config 3B (n=12) | – / 194 / – | – / 8.3 / – | – / 1.9 / – | – / 60 / – | – / 11.3 / – |
| SMa 7 GHz, 1299 m (n=12) | – / 65 / – | – / 3.9 / – | – / 1.5 / – | – / 25 / – | – / 14 / – |

For the 7 GHz rows only the median (p50) is shown. Additional p50 values:

- Coupling loss, config 1: UMi 6 GHz −105.8; UMa 6 GHz −118.6.
- Wideband SIR, config 1: UMi 3.7; UMa 5.8.

**Rel-14 full calibration (R1-165975, via QuaDRiGa `*38900` arrays; v14.0.0 parameters)** [V,
local file]. Format p5 / p50 / p95 at 6 GHz:

| Scenario | DS (ns) | ASD (°) | ZSD (°) | ASA (°) | ZSA (°) |
|---|---|---|---|---|---|
| UMi | 27.7 / 196.6 / 787.5 | 19.7 (p50) | 0.87 (p50) | 60.3 (p50) | 11.1 (p50) |
| UMa | 48.5 / 230.9 / 956.3 | 20.2 (p50) | 1.93 (p50) | 62.8 (p50) | 12.9 (p50) |
| InH | 15.8 (p50) | 35.3 (p50) | 11.2 (p50) | 46.2 (p50) | 11.8 (p50) |

**InF (R1-1909704)** [V; median of 4 companies, p10 / p50 / p90]. Coupling loss is in dB, DS in
ns, angles in degrees.

| Sub-scenario | Coupling loss | DS | ASD | ZSD | ASA | ZSA |
|---|---|---|---|---|---|---|
| SL 3.5 GHz | −66.4 / −60.5 / −51.8 | 94 / 174 / 299 | 11 / 29 / 68 | 6 / 16 / 38 | 11 / 33 / 78 | 6.5 / 18 / 40 |
| SL 28 GHz | −83.7 / −77.7 / −69.1 | – | – | – | – | – |
| DL 3.5 GHz | −81.7 / −73.1 / −60.3 | 99 / 171 / 299 | 22 / 39 / 76 | 7 / 17 / 45 | 23 / 56 / 105 | 11 / 26 / 47 |
| DL 28 GHz | −100.0 / −91.3 / −78.5 | – | – | – | – | – |
| SH 3.5 GHz | −74.3 / −69.0 / −62.2 | 120 / 222 / 393 | 11 / 28 / 65 | 6 / 16 / 38 | 10 / 29 / 73 | 6 / 17 / 38 |
| SH 28 GHz | −91.7 / −86.2 / −79.4 | – | – | – | – | – |
| DH 3.5 GHz | −68.8 / −64.4 / −59.3 | 80 / 141 / 242 | 20 / 38 / 74 | 7 / 17 / 40 | 21 / 52 / 105 | 11 / 23 / 42 |
| DH 28 GHz | −86.9 / −82.3 / −76.9 | – | – | – | – | – |

Geometry p50 values (with noise): SL 2.1, DL 5.3, SH 0.0, DH 0.0 dB. ASA values above 100° imply
non-circular or other AS definitions at some companies [I].

### 4.3 What the local QuaDRiGa-NG tutorials use

- `calib_3GPP_ref_data.m` [V]:
  - `cl36873a/b`, `gf36873a`, `zod36873a`, `sinr36873b`, `zsb/zsa36873b`, `sv*36873b` come from
    the 36.873 phases 1 and 2.
  - `cl38900a` and `gf38900a` come from **R1-165974**.
  - `cl38900b`, `sinr38900`, `ds/asd/zsd/asa/zsa38900` and `sv1/sv2/svR_38900` come from
    **R1-165975**.
  - All are Mean-column 5:5:95 percentiles based on 38.900 v14.0.0.
  - There is **no** InF, Rel-19 (6/7 GHz v19, SMa) or ISAC reference data.
- `calib_3GPP_38901_large_scale.m` and `calib_3GPP_38901_full.m` [V]:
  - Both cite "TR 38.901 V14.1.0".
  - Both note mismatches for UMa and InH, caused by the 38.900→38.901 parameter changes.
  - Large-scale script: 2000 UEs, 19 sites × 3 sectors, ISD 200/500/20 m, 80 % indoor, floors per
    36.873. Its 2000-UE sample is small for a 5 % tail [I].
  - Full script: also does singular-value metrics for config 2.

### 4.4 Proposed calibration test set for the new library

Notes for running it:

- Run each test as a system-level drop through the QuaDRiGa-NG tutorials, with the large-scale
  functions swapped to the new C++ library. Use ≥ 20,000 serving links per case [I].
- Compare p5/p50/p95 against the median across companies.
- Acceptance [I]: inside the min..max envelope across companies at every percentile, and within
  1 dB (coupling loss and geometry) or 10 % (spreads) of the median at p50.

| ID | Scenario / frequency | Layout | Metric | Reference | Tests |
|---|---|---|---|---|---|
| T1 | UMa, UMi 6 GHz | Table 7.8-1; 19×3, ISD 500/200, 80 % indoor, 50/50 O2I | Coupling loss; SIR without noise | Rel-19 `Calibration_LS_v31` UMi-6GHz/UMa-6GHz (final R1-2506406); compare also with R1-165974 to see the drift since v14 | PL, LOS probability, O2I, SF, pattern |
| T2 | UMa, UMi 7 GHz | Tables 7.8-1 and 7.8-1A | same | LS v31 UMi-7GHz/UMa-7GHz | Frequency dependence inside FR3 |
| T3 | SMa 7 GHz, ISD 1299 and 1732 m | Table 7.8-1A (mechanical tilt 95°/92°, 49 dBm, residential/commercial, car, low-loss A) | same | LS v31 SMa sheets; use the median because there are outliers | SMa PL/LOS probability, O2I low-loss A, car penetration |
| T4 | InH open office 6/30/70 GHz | Table 7.8-1 | Coupling loss; geometry | R1-165974 (only source; check the InH PL/LOS changes since v14 first) | InH PL and LOS probability |
| T5 | InF-SL/DL/SH/DH 3.5/28 GHz | Table 7.8-7 | Coupling loss; geometry; optionally DS/AS | R1-1909704 (4 companies; note the hall-height ambiguity) | InF PL, clutter LOS probability, InF LSPs |
| T6 | UMa, UMi 6 GHz, full, config 1 | Table 7.8-2 with the 7.8-2A additions | DS/ASD/ZSD/ASA/ZSA serving link; coupling loss; SIR | `Calibration_Full` (final R1-2506406) | LSP generation, cross-correlation, ZSD law. Coupling loss and SIR also check the ZOD offset indirectly. |
| T7 | UMa, UMi, SMa 7 GHz, full, config 3B | Table 7.8-2A | same | same | Rel-19 LSP tables at 7 GHz |
| U1 | Unit tests (not 3GPP) | – | PL, P_LOS, O2I, LSP mean/std, μ_offset,ZOD at fixed (d2D, hUT, fc) | TR v19.5.0 formulas | Exact; covers CR 0043, which postdates all calibrations |

## 5. Recommendations

1. **Use the Rel-19 v19.0.0 calibration (R1-2506406) as the primary reference for UMa/UMi, and add
   SMa at 7 GHz.**
   - It is the only calibration made with the current parameter tables, including the 6 GHz rerun
     "with updated channel modeling".
   - Download `TSGR1_122/Docs/R1-2506406.zip` (>10 MB) in a browser and diff it against the draft
     `Calibration_LS_v31` and `Calibration_Full_v34`.
   - Keep the 2016 R1-165974 curves only as a historical comparison. The QuaDRiGa ref data derives
     from them and is outdated for UMa and InH.
2. **Extend `calib_3GPP_ref_data.m` and the tutorials:**
   - Add the Rel-19 LS/full 6/7 GHz and SMa curves.
   - Add the InF curves (R1-1909704).
   - Store the per-company min..max envelope as well as the median, because company spreads are
     ±2–5 dB.
   - Raise the UE count well above 2000.
3. **Test ZOD offset and the post-v19.0.0 changes, including CR 0043, with deterministic unit
   tests against TR v19.5.0.** No calibration metric isolates them, and no calibration was rerun
   after v19.0.0. Treat a residual difference against the calibration curves in UMa NLOS
   coupling loss and SIR as possibly caused by these later corrections.

## 6. Not verified / open items

- Contents of R1-165975, R1-2504791, R1-2506406, R1-2506616 and R1-2509126. They are larger than
  the fetch limit; only their existence was confirmed. The values I report for Rel-19 come from
  **draft** workbooks in the public drafts folder.
- Number of companies in the 2016 full calibration (R1-165975), and the final company counts in
  R1-2506406 and R1-2509126.
- RAN1#123 location and dates. The date of R1-165975 (assumed RAN1#85 from its folder).
- Whether any Rel-20 (6G study, FS_Sensing_NR_Ph2) work defines new channel-model calibration. I
  found none.
- Why the InF hall height differs between the e-mail discussion (25/15 m) and TR Table 7.8-7
  (10 m), and which value the companies actually used.
- Whether the Rel-19 changes to InH or InF parameters invalidate the 2016 InH and 2019 InF
  references. This is not checked.
- Secondary claims (blog 6ghow.com; arXiv 2602.07623, 2606.07328, 2608.15283, 2610.05694) were
  not checked against primary documents, except where noted.

### Sources

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
