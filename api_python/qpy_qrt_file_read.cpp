// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

#include <thread>
#include <mutex>
#include <condition_variable>

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# qrt_file_read
Read ray-tracing CIR data from a QRT file

- Reads channel impulse response data from QRT files
- A file read cache is initialized once and reused across all requested snapshots, which
  significantly speeds up multi-snapshot reads
- File reading and the conversion to numpy arrays are overlapped: a background thread reads the
  next snapshot into a double buffer while the main thread converts the previous one
- If `downlink = True`, origin is TX and destination is RX; if `False`, the roles are swapped
- Per-snapshot outputs are returned as lists with one entry per requested snapshot

## Usage:
```
center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, path_length, M, aod, eod, aoa, eoa, \
    path_coord, no_int, coord, interact_type = quadriga_lib.channel.qrt_file_read( fn, cir, orig, downlink, normalize_M )
```

## Inputs:
- **`fn`** — Path to the QRT file
- **`cir`** — Snapshot indices to read; `(n_out,)` or `None`; if `None` or empty, all snapshots are read; default: `None`
- **`orig`** — Origin index (origin = TX for downlink); scalar; default: 0
- **`downlink`** — If `True`, origin=TX and destination=RX; if `False`, the roles are swapped; default: `True`
- **`normalize_M`** — Controls `M` and `path_gain` scaling; 0 = values as stored in the QRT file with
  `path_gain` = -PL; 1 = `M` columns scaled to max power 1 and `path_gain` = -PL minus material losses; default: 1
  - v4/v5 (EM):    FSPL = 32.45 + 20·log10(f_GHz) + 20·log10(d_m)  [dB]
  - v6 (scalar):   20·log10(d_m) + α(f)·d_m  [dB], with α from ISO 9613-1 at T=20°C, RH=50%, p=1 atm<br><br>
    | `normalize_M` | `M`                   | `path_gain`                      |
    | :-----------: | :-------------------: | :------------------------------: |
    | 0             | As stored in QRT file | -PL                              |
    | 1             | Max column power = 1  | -PL minus material losses        |

## Outputs:
- **`center_freq`** — Center frequencies in Hz; `(n_freq,)`
- **`tx_pos`** — Transmitter positions in Cartesian coordinates; `(3, n_out)`
- **`tx_orientation`** — Transmitter orientations as Euler angles (bank, tilt, heading); `(3, n_out)`
- **`rx_pos`** — Receiver positions in Cartesian coordinates; `(3, n_out)`
- **`rx_orientation`** — Receiver orientations as Euler angles (bank, tilt, heading); `(3, n_out)`
- **`fbs_pos`** — First-bounce scatterer positions; list of length `n_out`; entries `(3, n_path)`
- **`lbs_pos`** — Last-bounce scatterer positions; list of length `n_out`; entries `(3, n_path)`
- **`path_gain`** — Path gain in linear scale; list of length `n_out`; entries `(n_path, n_freq)`
- **`path_length`** — Absolute path length from TX to RX phase center; list of length `n_out`; entries `(n_path,)`
- **`M`** — Polarization transfer matrix, stored as interleaved real/imaginary pairs; list of length `n_out`; entries `(8, n_path, n_freq)`, or `(2, n_path, n_freq)` for v6 files
- **`aod`** — Departure azimuth angles; list of length `n_out`; entries `(n_path,)`
- **`eod`** — Departure elevation angles; list of length `n_out`; entries `(n_path,)`
- **`aoa`** — Arrival azimuth angles; list of length `n_out`; entries `(n_path,)`
- **`eoa`** — Arrival elevation angles; list of length `n_out`; entries `(n_path,)`
- **`path_coord`** — Interaction coordinates per path; list of length `n_out`; each entry is a list of length `n_path` with arrays `(3, n_interact + 2)`
- **`no_int`** — Number of mesh interactions per path (0 indicates LOS); list of length `n_out`; entries `(n_path,)`
- **`coord`** — Interaction coordinates concatenated across paths; list of length `n_out`; entries `(3, sum(no_int))`
- **`interact_type`** — Interaction type codes concatenated across paths (matching `coord`); list of length `n_out`;
  entries `(sum(no_int),)`. Empty for v4 legacy files.

## See also:
- [[generate]] (for generating antenna arrays)
- [[get_channels_planar]] (for embedding antennas using departure and arrival angles)
- [[get_channels_spherical]] (for embedding antennas using FBS/LBS positions)
- [[get_channels_multifreq]] (for multi-frequency antenna embedding)
MD!*/

// Per-snapshot read buffer, two of these are used as a double buffer
struct qd_qrt_slot
{
    arma::vec tx_pos, tx_orientation, rx_pos, rx_orientation;
    arma::mat fbs_pos, lbs_pos, path_gain;
    arma::vec path_length, aod, eod, aoa, eoa;
    arma::cube M;
    std::vector<arma::mat> path_coord;
    arma::u32_vec no_int;
    arma::fmat coord;
    std::vector<uint8_t> interact_type;
};

py::tuple qrt_file_read(const std::string &fn,
                        py::handle cir,
                        arma::uword orig,
                        bool downlink,
                        int normalize_M)
{
    // Open file stream and initialize the read cache (reused across all snapshots)
    std::ifstream stream(fn, std::ios::in | std::ios::binary);
    auto cache = quadriga_lib::qrt_read_cache_init(fn, &stream);

    // Resolve snapshot indices (None or empty = read all snapshots)
    arma::uvec i_cir_a = qd_python_numpy2arma_Col<arma::uword>(cir, true);
    if (i_cir_a.empty())
        i_cir_a = arma::regspace<arma::uvec>(0, arma::uword(cache.no_cir - 1));
    else if (arma::any(i_cir_a >= arma::uword(cache.no_cir)))
        throw std::out_of_range("CIR index exceeds number of CIRs in file.");
    arma::uword no_out = i_cir_a.n_elem;

    // Center frequencies in Hz (taken from the cache, identical for all snapshots)
    arma::uword no_freq = (arma::uword)cache.no_freq;
    arma::vec center_freq;
    auto center_freq_py = qd_python_init_output(no_freq, &center_freq);
    for (arma::uword i_freq = 0; i_freq < no_freq; ++i_freq)
        center_freq[i_freq] = (cache.version == 6) ? (double)cache.freq[i_freq] : 1.0e9 * (double)cache.freq[i_freq];

    // Fixed-size outputs, one column per requested snapshot
    arma::mat tx_pos, tx_orientation, rx_pos, rx_orientation;
    auto tx_pos_py = qd_python_init_output(3, no_out, &tx_pos);
    auto tx_orientation_py = qd_python_init_output(3, no_out, &tx_orientation);
    auto rx_pos_py = qd_python_init_output(3, no_out, &rx_pos);
    auto rx_orientation_py = qd_python_init_output(3, no_out, &rx_orientation);

    // Per-snapshot outputs (one list entry per requested snapshot)
    py::list fbs_pos_py, lbs_pos_py, path_gain_py, path_length_py, M_py;
    py::list aod_py, eod_py, aoa_py, eoa_py, path_coord_py, no_int_py, coord_py, interact_type_py;

    // Double buffer, the reader thread fills one slot while the main thread converts the other
    qd_qrt_slot slot[2];
    std::mutex mtx;
    std::condition_variable cv;
    arma::uword n_produced = 0ULL; // Number of slots filled by the reader
    arma::uword n_consumed = 0ULL; // Number of slots converted by the main thread
    bool reader_failed = false;
    std::string reader_error;

    // Reader thread: owns the file stream and the cache, never touches Python objects
    std::thread reader([&]()
                       {
        for (arma::uword i = 0ULL; i < no_out; ++i)
        {
            // Wait until a free slot is available
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&] { return n_produced - n_consumed < 2ULL; });
            }

            try
            {
                qd_qrt_slot &s = slot[i % 2ULL];
                quadriga_lib::qrt_file_read<double>(fn, i_cir_a[i], orig, downlink, nullptr,
                                                    &s.tx_pos, &s.tx_orientation, &s.rx_pos, &s.rx_orientation,
                                                    &s.fbs_pos, &s.lbs_pos, &s.path_gain, &s.path_length, &s.M,
                                                    &s.aod, &s.eod, &s.aoa, &s.eoa, &s.path_coord, normalize_M,
                                                    &s.no_int, &s.coord, &s.interact_type, &stream, &cache);
            }
            catch (const std::exception &e)
            {
                std::lock_guard<std::mutex> lk(mtx);
                reader_failed = true;
                reader_error = "Snapshot " + std::to_string(i) + ": " + e.what();
                cv.notify_all();
                return;
            }

            {
                std::lock_guard<std::mutex> lk(mtx);
                ++n_produced;
            }
            cv.notify_all();
        } });

    // Main thread: converts the filled slots to numpy, holds the GIL
    for (arma::uword i_out = 0ULL; i_out < no_out; ++i_out)
    {
        // Wait for the next filled slot, release the GIL while waiting
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [&] { return n_produced > n_consumed || reader_failed; });
            if (reader_failed)
                break;
        }

        qd_qrt_slot &s = slot[i_out % 2ULL];

        tx_pos.col(i_out) = s.tx_pos;
        tx_orientation.col(i_out) = s.tx_orientation;
        rx_pos.col(i_out) = s.rx_pos;
        rx_orientation.col(i_out) = s.rx_orientation;

        fbs_pos_py.append(qd_python_copy2numpy(&s.fbs_pos));
        lbs_pos_py.append(qd_python_copy2numpy(&s.lbs_pos));
        path_gain_py.append(qd_python_copy2numpy(&s.path_gain));
        path_length_py.append(qd_python_copy2numpy(&s.path_length));
        M_py.append(qd_python_copy2numpy(&s.M));
        aod_py.append(qd_python_copy2numpy(&s.aod));
        eod_py.append(qd_python_copy2numpy(&s.eod));
        aoa_py.append(qd_python_copy2numpy(&s.aoa));
        eoa_py.append(qd_python_copy2numpy(&s.eoa));
        path_coord_py.append(qd_python_copy2list(&s.path_coord));
        no_int_py.append(qd_python_copy2numpy(&s.no_int));
        coord_py.append(qd_python_copy2numpy(&s.coord));

        arma::Col<arma::u8> it_view(s.interact_type.data(), s.interact_type.size(), false, true);
        interact_type_py.append(qd_python_copy2numpy(&it_view));

        // Release the slot back to the reader
        {
            std::lock_guard<std::mutex> lk(mtx);
            ++n_consumed;
        }
        cv.notify_all();
    }

    // Join the reader before touching the stream or reporting errors
    {
        py::gil_scoped_release release;
        reader.join();
    }

    stream.close();

    if (reader_failed)
        throw std::runtime_error(reader_error.c_str());

    // Return tuple
    return py::make_tuple(center_freq_py, tx_pos_py, tx_orientation_py, rx_pos_py, rx_orientation_py,
                          fbs_pos_py, lbs_pos_py, path_gain_py, path_length_py, M_py,
                          aod_py, eod_py, aoa_py, eoa_py, path_coord_py, no_int_py, coord_py,
                          interact_type_py);
}

// pybind11 declaration:
// m.def("qrt_file_read", &qrt_file_read,
//       py::arg("fn"),
//       py::arg("i_cir") = py::none(),
//       py::arg("i_orig") = 0,
//       py::arg("downlink") = true,
//       py::arg("normalize_M") = 1);