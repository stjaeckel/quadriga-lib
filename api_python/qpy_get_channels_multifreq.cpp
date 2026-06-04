// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# get_channels_multifreq
Compute channel coefficients for spherical waves across multiple frequencies

- Multi-frequency extension of [[get_channels_spherical]] with frequency-dependent antenna patterns, path gains, and Jones matrices
- Geometry (angles, element delays, LOS detection) is computed once and reused across all output frequencies
- Aligns four frequency grids: TX array (`ant_tx['center_freq']`), RX array (`ant_rx['center_freq']`), input samples (`freq_in`), and output (`freq_out`)
- Antenna pattern fields may be 3D (single-frequency, clamped for all output frequencies) or 4D (multi-frequency, 4th axis = frequency)
- TX/RX patterns are interpolated per output frequency via SLERP with linear fallback
- `path_gain` is interpolated linearly; `M` is interpolated via SLERP per complex entry pair to preserve phase
- Coupling matrices are interpolated across frequencies (SLERP for complex pairs); pass 3D coupling `(n_elem, n_ports, n_freq)` for per-frequency coupling
- Extrapolation clamps to the nearest frequency entry on all four grids
- `M` accepts 8 rows (full polarimetric: ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH) or 2 rows (scalar pressure: ReVV, ImVV only)
- `propagation_speed` supports EM (speed of light, default) and acoustic (343 m/s) simulations
- `n_path_out = n_path + 1` if `add_fake_los_path` else `n_path`
- `complex=True` returns one combined complex coefficient array `coeff`; `complex=False` (default) returns
  separate real `coeff_re` and `coeff_im` via a zero-copy fast path

## Usage:
```
coeff_re, coeff_im, delay = quadriga_lib.arrayant.get_channels_multifreq( ant_tx, ant_rx, fbs_pos, \
    lbs_pos, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, freq_in, \
    freq_out, use_absolute_delays, add_fake_los_path, propagation_speed )

coeff, delay = quadriga_lib.arrayant.get_channels_multifreq( ..., complex=True )
```

## Inputs:
- **`ant_tx`** — Multi-frequency TX arrayant dict; pattern fields 3D `(n_el, n_az, n_elem)` or 4D `(n_el, n_az, n_elem, n_freq)`; see [[generate]]
- **`ant_rx`** — RX arrayant dict; same format as `ant_tx`
- **`fbs_pos`** — First-bounce scatterer positions; `(3, n_path)`
- **`lbs_pos`** — Last-bounce scatterer positions; `(3, n_path)`
- **`path_gain`** — Linear-scale path gains, one column per input frequency; `(n_path, n_freq_in)`
- **`path_length`** — Absolute TX-to-RX path lengths; `(n_path,)`
- **`M`** — Polarization transfer matrix, interleaved Re/Im; `(8, n_path, n_freq_in)` full pol (VV, VH, HV, HH)
  or `(2, n_path, n_freq_in)` scalar pressure (ReVV, ImVV only)
- **`tx_pos`** — Transmitter position; `(3,)`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`rx_pos`** — Receiver position; `(3,)`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`freq_in`** — Input sample frequencies for `path_gain` and `M`; `(n_freq_in,)`
- **`freq_out`** — Target output frequencies; `(n_freq_out,)`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`add_fake_los_path`** — If `True`, prepends a zero-power LOS path when none is present; default: `False`
- **`propagation_speed`** — Wave speed in m/s; use ~343.0 for acoustics; default: `299792458.0`
- **`complex`** — If `True`, combine coefficients into a single complex array `coeff`; if `False`, return
  separate `coeff_re` and `coeff_im`; default: `False`

## Outputs:
- **`coeff_re`** — Real part of channel coefficients (`complex=False`); `(n_ports_rx, n_ports_tx, n_path_out, n_freq_out)`
- **`coeff_im`** — Imaginary part of channel coefficients (`complex=False`); same shape as `coeff_re`
- **`coeff`** — Complex channel coefficients (`complex=True`), replaces `coeff_re`/`coeff_im`; same shape
- **`delay`** — Propagation delays in seconds; `(n_ports_rx, n_ports_tx, n_path_out, n_freq_out)`

## See also:
- [[get_channels_spherical]] (single-frequency equivalent)
- [[generate_speaker]] (acoustic source construction)
MD!*/

py::tuple get_channels_multifreq(const py::dict &ant_tx,
                                 const py::dict &ant_rx,
                                 const py::array_t<double> &fbs_pos,
                                 const py::array_t<double> &lbs_pos,
                                 const py::array_t<double> &path_gain,
                                 const py::array_t<double> &path_length,
                                 const py::array_t<double> &M,
                                 const py::array_t<double> &tx_pos,
                                 const py::array_t<double> &tx_orientation,
                                 const py::array_t<double> &rx_pos,
                                 const py::array_t<double> &rx_orientation,
                                 const py::array_t<double> &freq_in,
                                 const py::array_t<double> &freq_out,
                                 const bool use_absolute_delays,
                                 const bool add_fake_los_path,
                                 const double propagation_speed,
                                 const bool complex)
{
    // Parse antenna dictionaries (3D single-freq or 4D multi-freq handled internally)
    auto ant_tx_vec = qd_python_dict2arrayant_multi(ant_tx, true);
    auto ant_rx_vec = qd_python_dict2arrayant_multi(ant_rx, true);

    // Parse remaining input arguments
    const auto fbs_pos_a = qd_python_numpy2arma_Mat(fbs_pos, true);
    const auto lbs_pos_a = qd_python_numpy2arma_Mat(lbs_pos, true);
    const auto path_gain_a = qd_python_numpy2arma_Mat(path_gain, true);
    const auto path_length_a = qd_python_numpy2arma_Col(path_length, true);
    const auto M_a = qd_python_numpy2arma_Cube(M, true);
    const auto tx_pos_a = qd_python_numpy2arma_Col(tx_pos, true, false, "tx_pos", 3);
    const auto tx_orientation_a = qd_python_numpy2arma_Col(tx_orientation, true, false, "tx_orientation", 3);
    const auto rx_pos_a = qd_python_numpy2arma_Col(rx_pos, true, false, "rx_pos", 3);
    const auto rx_orientation_a = qd_python_numpy2arma_Col(rx_orientation, true, false, "rx_orientation", 3);
    const auto freq_in_a = qd_python_numpy2arma_Col(freq_in, true);
    const auto freq_out_a = qd_python_numpy2arma_Col(freq_out, true);

    // Extract scalar values
    double Tx = tx_pos_a.at(0), Ty = tx_pos_a.at(1), Tz = tx_pos_a.at(2);
    double Tb = tx_orientation_a.at(0), Tt = tx_orientation_a.at(1), Th = tx_orientation_a.at(2);
    double Rx = rx_pos_a.at(0), Ry = rx_pos_a.at(1), Rz = rx_pos_a.at(2);
    double Rb = rx_orientation_a.at(0), Rt = rx_orientation_a.at(1), Rh = rx_orientation_a.at(2);

    // Derived output sizes
    arma::uword n_ports_tx = ant_tx_vec[0].n_ports();
    arma::uword n_ports_rx = ant_rx_vec[0].n_ports();
    arma::uword n_path = add_fake_los_path ? fbs_pos_a.n_cols + 1 : fbs_pos_a.n_cols;
    arma::uword n_freq_out = freq_out_a.n_elem;

    // C++ core needs at least one scatterer (the MEX guard); otherwise return zero-filled output
    bool has_path = (fbs_pos_a.n_cols != 0);

    std::vector<arma::cube> coeff_re, coeff_im, delay;

    if (complex)
    {
        // Complex path: C++ sizes plain owning cubes, then stack2numpy interleaves Re/Im into one
        // complex array. init_output cannot back a complex buffer, so this path copies the result once.
        if (has_path)
            quadriga_lib::get_channels_multifreq<double>(ant_tx_vec, ant_rx_vec,
                                                         Tx, Ty, Tz, Tb, Tt, Th,
                                                         Rx, Ry, Rz, Rb, Rt, Rh,
                                                         fbs_pos_a, lbs_pos_a, path_gain_a, path_length_a, M_a,
                                                         freq_in_a, freq_out_a,
                                                         coeff_re, coeff_im, delay,
                                                         use_absolute_delays, add_fake_los_path, propagation_speed);
        else // no scatterer: fabricate zero-filled outputs
        {
            coeff_re.assign(n_freq_out, arma::cube(n_ports_rx, n_ports_tx, n_path, arma::fill::zeros));
            coeff_im = coeff_re;
            delay = coeff_re;
        }

        auto coeff_py = qd_python_stack2numpy<double, std::complex<double>>(&coeff_re, &coeff_im);
        auto delay_py = qd_python_stack2numpy(&delay);
        return py::make_tuple(coeff_py, delay_py);
    }

    // Allocate 4D outputs, frequency on the 4th axis (zero-copy, written in place by C++)
    auto coeff_re_py = qd_python_init_output(n_ports_rx, n_ports_tx, n_path, n_freq_out, &coeff_re);
    auto coeff_im_py = qd_python_init_output(n_ports_rx, n_ports_tx, n_path, n_freq_out, &coeff_im);
    auto delay_py = qd_python_init_output(n_ports_rx, n_ports_tx, n_path, n_freq_out, &delay);

    // Call C++ library function; skip when no real path exists (outputs stay zero-filled)
    if (has_path)
        quadriga_lib::get_channels_multifreq<double>(ant_tx_vec, ant_rx_vec,
                                                     Tx, Ty, Tz, Tb, Tt, Th,
                                                     Rx, Ry, Rz, Rb, Rt, Rh,
                                                     fbs_pos_a, lbs_pos_a, path_gain_a, path_length_a, M_a,
                                                     freq_in_a, freq_out_a,
                                                     coeff_re, coeff_im, delay,
                                                     use_absolute_delays, add_fake_los_path, propagation_speed);
    else // no scatterer: init_output memory is not guaranteed zeroed, so fill explicitly
        for (arma::uword i = 0; i < n_freq_out; ++i)
        {
            coeff_re[i].zeros();
            coeff_im[i].zeros();
            delay[i].zeros();
        }

    return py::make_tuple(coeff_re_py, coeff_im_py, delay_py);
}

// pybind11 declaration:
// m.def("get_channels_multifreq", &get_channels_multifreq,
//       py::arg("ant_tx"),
//       py::arg("ant_rx"),
//       py::arg("fbs_pos"),
//       py::arg("lbs_pos"),
//       py::arg("path_gain"),
//       py::arg("path_length"),
//       py::arg("M"),
//       py::arg("tx_pos"),
//       py::arg("tx_orientation"),
//       py::arg("rx_pos"),
//       py::arg("rx_orientation"),
//       py::arg("freq_in"),
//       py::arg("freq_out"),
//       py::arg("use_absolute_delays") = false,
//       py::arg("add_fake_los_path") = false,
//       py::arg("propagation_speed") = 299792458.0,
//       py::arg("complex") = false);
