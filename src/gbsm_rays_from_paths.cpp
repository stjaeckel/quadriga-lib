// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_gbsm.hpp"
#include "quadriga_gbsm_helpers.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# gbsm_rays_from_paths
Ray data of one link from path data (positions, gains, lengths, polarization), e.g. read with qrt_file_read

- Fills a [[gbsm_rays]] object in place (`fallback` cleared) from path data in the names and order of
  [[qrt_file_read]], read as `qrt_file_read<double>` with `downlink = true` and `normalize_M = 1` (an uplink read gives
  swapped ends and a conjugated `M`, which this function does not undo; `normalize_M = 0` leaves only the free-space
  gain in `power`)
- `delay` = (path length - Tx-Rx distance) / c, `power` = path gain. Angles point from the Tx to the first of
  `fbs_pos`, `lbs_pos`, `rx_pos` that differs from it and from the Rx to the first of `lbs_pos`, `fbs_pos`, `tx_pos`;
  GR angles to the image of the other end (z mirrored); 0 if `tx_pos` = `rx_pos`
- QRT stores no path types or taps. Without `path_type`: LOS where FBS = LBS on the direct line (0.1 mm), else
  NLOS. A GR path needs `path_type`: for files of [[qrt_paths_from_scatterers]] pass the `no_int` of [[qrt_file_read]]
  (interaction count = path type); for ray-traced files the user sets it (values > 2 throw)
- The GR point (`lbs_pos`) must be within 1 mm of z = 0

## Declaration:
```
void quadriga_lib::gbsm_rays_from_paths(
    const arma::vec &center_frequency,
    const arma::vec &tx_pos,
    const arma::vec &rx_pos,
    const arma::mat &fbs_pos,
    const arma::mat &lbs_pos,
    const arma::mat &path_gain,
    const arma::vec &path_length,
    const arma::cube &M,
    gbsm_rays &rays,
    const std::string &method = "quadriga",
    const arma::u32_vec *path_type = nullptr,
    const arma::u32_vec *tap = nullptr,
    const arma::vec *gr_eps_re = nullptr,
    const arma::vec *gr_eps_im = nullptr);
```

## Inputs:
- **`center_frequency`** — Frequencies in [Hz]; `[n_freq]`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in [m]; `[3]`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in [m]; `[3, n_path]`
- **`path_gain`** — Path gains (linear); `[n_path, n_freq]`
- **`path_length`** — Path lengths in [m]; `[n_path]`
- **`M`** — Polarization transfer matrices; `[8, n_path, n_freq]`
- **`method`** *(optional)* — Value of `rays.method`; sets the GR power law and the default mobility
- **`path_type`** *(optional)* — Path type, 0 = LOS, 1 = GR, 2 = NLOS, at most one LOS and one GR; `[n_path]`
- **`tap`** *(optional)* — Tap index; omitted: LOS 0, GR 3, the others 6, 9, ... in path order; `[n_path]`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity, given together; needed only where a GR ray is
  recomputed, i.e. in the mobility `drift` at moved ends; `[n_freq]`

## Output:
- **`rays`** — Ray data with `n_ray = n_path`, see [[gbsm_rays]]; use with `link_gain_dB = 0` and without `kf_ratio`

## See also:
- [[qrt_file_read]] (source of the path data, `downlink = true`)
- [[qrt_paths_from_scatterers]] (write the paths of [[ssf_paths]] to a QRT file)
- [[ssf_paths]] (paths at moved positions)
- [[gbsm_single_link_channel]] (channel of the link)
MD!*/

void quadriga_lib::gbsm_rays_from_paths(const arma::vec &center_frequency,
                                        const arma::vec &tx_pos,
                                        const arma::vec &rx_pos,
                                        const arma::mat &fbs_pos,
                                        const arma::mat &lbs_pos,
                                        const arma::mat &path_gain,
                                        const arma::vec &path_length,
                                        const arma::cube &M,
                                        gbsm_rays &rays,
                                        const std::string &method,
                                        const arma::u32_vec *path_type,
                                        const arma::u32_vec *tap,
                                        const arma::vec *gr_eps_re,
                                        const arma::vec *gr_eps_im)
{
    const arma::uword n_freq = center_frequency.n_elem, n_path = fbs_pos.n_cols;
    if (n_freq == 0)
        throw std::invalid_argument("Input 'center_frequency' cannot be empty.");
    if (!center_frequency.is_finite() || arma::any(center_frequency <= 0.0))
        throw std::invalid_argument("Input 'center_frequency' must be finite and > 0.");
    auto check_pos = [](const arma::vec &x, const std::string &name)
    {
        if (x.n_elem != 3)
            throw std::invalid_argument("Input '" + name + "' must have 3 elements.");
        if (!x.is_finite())
            throw std::invalid_argument("Input '" + name + "' must be finite.");
    };
    check_pos(tx_pos, "tx_pos"), check_pos(rx_pos, "rx_pos");
    if (fbs_pos.n_rows != 3 || lbs_pos.n_rows != 3 || lbs_pos.n_cols != n_path)
        throw std::invalid_argument("Inputs 'fbs_pos' and 'lbs_pos' must have size [3, n_path].");
    if (!fbs_pos.is_finite() || !lbs_pos.is_finite())
        throw std::invalid_argument("Inputs 'fbs_pos' and 'lbs_pos' must be finite.");
    if (path_gain.n_rows != n_path || path_gain.n_cols != n_freq)
        throw std::invalid_argument("Input 'path_gain' must have size [n_path, n_freq].");
    if (!path_gain.is_finite() || arma::any(arma::vectorise(path_gain) < 0.0))
        throw std::invalid_argument("Input 'path_gain' must be finite and >= 0.");
    if (path_length.n_elem != n_path)
        throw std::invalid_argument("Input 'path_length' must have n_path elements.");
    if (!path_length.is_finite() || arma::any(path_length < 0.0))
        throw std::invalid_argument("Input 'path_length' must be finite and >= 0.");
    if (M.n_rows != 8 || M.n_cols != n_path || M.n_slices != n_freq)
        throw std::invalid_argument("Input 'M' must have size [8, n_path, n_freq].");
    if (!M.is_finite())
        throw std::invalid_argument("Input 'M' must be finite.");
    if (method != "38.901" && method != "38.901_MF" && method != "quadriga")
        throw std::invalid_argument("Input 'method' must be '38.901', '38.901_MF' or 'quadriga'.");

    const double *T = tx_pos.memptr(), *R = rx_pos.memptr(), d3D = qd_gbsm_dist(T, R);

    // Path types: given, or LOS where FBS = LBS on the direct line (the LOS tolerance 0.1 mm of get_channels_*)
    arma::u32_vec type(n_path);
    if (path_type)
    {
        if (path_type->n_elem != n_path)
            throw std::invalid_argument("Input 'path_type' must have n_path elements.");
        if (arma::any(*path_type > 2))
            throw std::invalid_argument("Values of 'path_type' must be 0 (LOS), 1 (GR) or 2 (NLOS).");
        if (arma::accu(*path_type == 0) > 1 || arma::accu(*path_type == 1) > 1)
            throw std::invalid_argument("Input 'path_type' can hold at most one LOS and one GR path.");
        type = *path_type;
    }
    else
    {
        for (arma::uword p = 0; p < n_path; ++p)
            type(p) = (qd_gbsm_dist(fbs_pos.colptr(p), lbs_pos.colptr(p)) <= 1.0e-4 && path_length(p) - d3D <= 1.0e-4)
                          ? 0
                          : 2;
        if (arma::accu(type == 0) > 1)
            throw std::invalid_argument("More than one path matches the LOS geometry; pass 'path_type'.");
    }
    if (tap && tap->n_elem != n_path)
        throw std::invalid_argument("Input 'tap' must have n_path elements.");
    if ((gr_eps_re == nullptr) != (gr_eps_im == nullptr))
        throw std::invalid_argument("Inputs 'gr_eps_re' and 'gr_eps_im' must both be given or both be omitted.");
    if (gr_eps_re)
    {
        if (gr_eps_re->n_elem != n_freq || gr_eps_im->n_elem != n_freq)
            throw std::invalid_argument("Inputs 'gr_eps_re' and 'gr_eps_im' must have n_freq elements.");
        if (!gr_eps_re->is_finite() || !gr_eps_im->is_finite() || arma::any(*gr_eps_re < 1.0) ||
            arma::any(*gr_eps_im < 0.0))
            throw std::invalid_argument("Ground permittivity needs 'gr_eps_re' >= 1 and 'gr_eps_im' >= 0.");
    }
    // 1 mm: float32 keeps z = 0 of a GR point for any x, y offset; a larger z means coordinates shifted in z
    for (arma::uword p = 0; p < n_path; ++p)
        if (type(p) == 1 && std::abs(lbs_pos(2, p)) > 1.0e-3)
            throw std::invalid_argument("The interaction point of the GR path must be on the ground (z = 0); "
                                        "shift local coordinates in x and y only.");

    rays.delay.set_size(n_path), rays.aod.set_size(n_path), rays.eod.set_size(n_path);
    rays.aoa.set_size(n_path), rays.eoa.set_size(n_path);
    const double R_img[3] = {R[0], R[1], -R[2]}, T_img[3] = {T[0], T[1], -T[2]};
    for (arma::uword p = 0; p < n_path; ++p)
    {
        const double *F = fbs_pos.colptr(p), *B = lbs_pos.colptr(p);
        rays.delay(p) = (path_length(p) - d3D) / 299792458.0;
        if (type(p) == 1) // GR: towards the image points, also defined for an end on the ground
            qd_gbsm_direction(T, {R_img}, rays.aod(p), rays.eod(p)),
                qd_gbsm_direction(R, {T_img}, rays.aoa(p), rays.eoa(p));
        else
            qd_gbsm_direction(T, {F, B, R}, rays.aod(p), rays.eod(p)),
                qd_gbsm_direction(R, {B, F, T}, rays.aoa(p), rays.eoa(p));
    }

    rays.method = method;
    rays.center_frequency = center_frequency, rays.tx_pos = tx_pos, rays.rx_pos = rx_pos;
    rays.power = path_gain, rays.M = M, rays.fbs_pos = fbs_pos, rays.lbs_pos = lbs_pos;
    rays.path_type = type;
    if (tap)
        rays.tap = *tap;
    else // One slot per path: LOS 0, GR 3, the others 6, 9, ... in path order
    {
        rays.tap.set_size(n_path);
        unsigned slot = 2; // slot of the next NLOS path
        for (arma::uword p = 0; p < n_path; ++p)
            rays.tap(p) = (type(p) == 0) ? 0 : (type(p) == 1) ? 3 : 3 * slot++;
    }
    if (gr_eps_re)
        rays.gr_eps_re = *gr_eps_re, rays.gr_eps_im = *gr_eps_im;
    else
    {
        if (!rays.gr_eps_re.empty())
            rays.gr_eps_re.reset();
        if (!rays.gr_eps_im.empty())
            rays.gr_eps_im.reset();
    }
    if (!rays.fallback.empty())
        rays.fallback.reset();
}
