// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Helpers shared by the MATLAB / Octave (api_mex) and Python (api_python) wrappers; they use no MATLAB or Python types

#ifndef qd_helpers_wrappers_hpp
#define qd_helpers_wrappers_hpp

#include "quadriga_arrayant.hpp"
#include "quadriga_channel.hpp"

#include <armadillo>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// Defaults of the optional arrayant fields after reading a MATLAB struct or Python dict: an empty 'element_pos' becomes
// zeros [3, n_elements], an empty coupling (both parts empty) identity [n_elements, n_elements], an empty
// 'coupling_im' zeros of the size of 'coupling_re'; given members stay unchanged. The library treats empty members
// the same way, so this changes only the arrays that the wrappers return.
static inline void qd_arrayant_fill_defaults(quadriga_lib::arrayant<double> &ant)
{
    const arma::uword n_elements = ant.e_theta_re.n_slices;
    if (ant.element_pos.is_empty())
        ant.element_pos.zeros(3, n_elements);
    if (ant.coupling_re.is_empty() && ant.coupling_im.is_empty())
        ant.coupling_re.eye(n_elements, n_elements);
    if (ant.coupling_im.is_empty())
        ant.coupling_im.zeros(ant.coupling_re.n_rows, ant.coupling_re.n_cols);
}

// Occupied slots of an HDF5 slot selection, for hdf5_read_channel_vec (used by the MEX and Python read wrappers)
// - fn, dims, channel_id: file name, layout {nx, ny, nz, nw} and channel ID per slot, as returned by
//   hdf5_read_layout; dims = 0 (missing file) throws std::runtime_error
// - ix, iy, iz, iw: 0-based indices per dimension (any order, repeats allowed); empty = full extent; an index outside
//   the layout throws std::invalid_argument
// - Returns the slots with a non-zero channel ID in column-major order (ix fastest), one per column; [4, n_slot]
static inline arma::u32_mat qd_hdf5_select_slots(const std::string &fn, const arma::u32_vec &dims,
                                                 const arma::u32_vec &channel_id, const arma::u32_vec &ix,
                                                 const arma::u32_vec &iy, const arma::u32_vec &iz,
                                                 const arma::u32_vec &iw)
{
    if (dims.n_elem != 4 || dims.at(0) == 0)
        throw std::runtime_error("File '" + fn + "' does not exist.");
    const arma::u32_vec *given[4] = {&ix, &iy, &iz, &iw};
    const char *names[4] = {"ix", "iy", "iz", "iw"};
    arma::u32_vec sel[4];
    for (int d = 0; d < 4; ++d)
    {
        if (given[d]->is_empty())
            sel[d] = arma::regspace<arma::u32_vec>(0, dims.at(d) - 1);
        else if (arma::any(*given[d] >= dims.at(d)))
            throw std::invalid_argument("Input '" + std::string(names[d]) + "' contains indices outside the storage layout (" +
                                        std::to_string(dims.at(d)) + " slots).");
        else
            sel[d] = *given[d];
    }
    const uint64_t nx = dims.at(0), ny = dims.at(1), nz = dims.at(2);
    std::vector<unsigned> slots;
    for (const unsigned w : sel[3])
        for (const unsigned z : sel[2])
            for (const unsigned y : sel[1])
                for (const unsigned x : sel[0])
                    if (channel_id.at((arma::uword)(x + nx * (y + ny * (z + nz * w)))) != 0)
                        slots.insert(slots.end(), {x, y, z, w});
    if (slots.empty())
        return arma::u32_mat(4, 0);
    return arma::u32_mat(slots.data(), 4, slots.size() / 4);
}

// ---- qrt_file_read with a byte limit shared by several calls (defined in qd_channel_qrt.cpp) ----
// - Declared here only, to keep the wrapper header out of the core sources; a signature that differs from the
//   definition fails at link time (the parameter types are part of the symbol name)
// - 'bytes_read': path-block bytes read so far by all calls; 'repeat': largest number of times one CIR occurs in the
//   whole list, computed by the first call that gets the whole list (0: not computed yet); nullptr: per call
// - Throws std::runtime_error when the bytes read exceed 'repeat' times the file size
template <typename dtype>
void qd_qrt_file_read_impl(const std::string &fn, const arma::uvec &i_cir, arma::uword i_orig, bool uplink,
                           arma::Col<dtype> *center_freq, arma::Mat<dtype> *tx_pos, arma::Mat<dtype> *tx_orientation,
                           arma::Mat<dtype> *rx_pos, arma::Mat<dtype> *rx_orientation,
                           std::vector<arma::Mat<dtype>> *fbs_pos, std::vector<arma::Mat<dtype>> *lbs_pos,
                           std::vector<arma::Mat<dtype>> *path_gain, std::vector<arma::Col<dtype>> *path_length,
                           std::vector<arma::Cube<dtype>> *polmat,
                           std::vector<arma::Col<dtype>> *aod, std::vector<arma::Col<dtype>> *eod,
                           std::vector<arma::Col<dtype>> *aoa, std::vector<arma::Col<dtype>> *eoa,
                           std::vector<std::vector<arma::Mat<dtype>>> *path_coord, int normalize_polmat,
                           std::vector<arma::u32_vec> *n_interact, std::vector<arma::fmat> *coord,
                           std::vector<std::vector<uint8_t>> *interact_type,
                           std::ifstream *file, const quadriga_lib::qrt_read_cache *cache,
                           size_t *bytes_read, size_t *repeat);

// ---- Chunked, double-buffered reading of QRT CIRs (qrt_file_read) ----
// - A reader thread reads the next chunk of CIRs while the calling thread converts the previous one, so that file
//   reading and conversion overlap; the peak memory is the converted output plus two chunks
// - The reader thread uses only the shared state on the heap (stream, cache, buffers), never MATLAB or Python objects
// - No std::exception_ptr and no std::condition_variable::wait: the MEX files must load with the older libstdc++ of
//   MATLAB (tests/check_mex_symbol_versions.sh); errors are passed as type and message, waits use wait_for

// Adds 'offset' to the position in a "Snapshot <k>: " prefix of a qrt_file_read message about one CIR of a chunk
static inline std::string qd_qrt_shift_prefix(const std::string &msg, arma::uword offset)
{
    const std::string p = "Snapshot ";
    if (offset == 0 || msg.compare(0, p.size(), p) != 0)
        return msg;
    const size_t end = msg.find_first_not_of("0123456789", p.size());
    if (end == p.size() || end == std::string::npos || msg.compare(end, 2, ": ") != 0)
        return msg;
    return p + std::to_string(std::stoull(msg.substr(p.size(), end - p.size())) + offset) + msg.substr(end);
}

// Read buffer of one chunk, per-CIR outputs of qrt_file_read
struct qd_qrt_chunk
{
    arma::uvec i_cir; // CIR indices of this chunk
    arma::mat tx_pos, tx_orientation, rx_pos, rx_orientation;
    std::vector<arma::mat> fbs_pos, lbs_pos, path_gain;
    std::vector<arma::vec> path_length, aod, eod, aoa, eoa;
    std::vector<arma::cube> polmat;
    std::vector<std::vector<arma::mat>> path_coord;
    std::vector<arma::u32_vec> n_interact;
    std::vector<arma::fmat> coord;
    std::vector<std::vector<uint8_t>> interact_type;
};

// State shared by the calling thread and the reader thread
struct qd_qrt_chunk_state
{
    std::ifstream stream;
    quadriga_lib::qrt_read_cache cache;
    size_t bytes_read = 0, repeat = 0; // byte limit shared by all chunks: one list of CIRs
    arma::uvec i_cir;
    arma::uword i_orig = 0, chunk_size = 1, n_chunk = 0;
    bool uplink = false;
    int normalize_polmat = 1, n_req = 0;

    qd_qrt_chunk slot[2];
    std::mutex mtx;
    std::condition_variable cv;
    arma::uword n_produced = 0, n_consumed = 0;
    bool stop = false;
    int error = 0; // Error of the reader: 0 none, 1 std::invalid_argument, 2 std::bad_alloc, 3 other (std::runtime_error)
    std::string error_msg;

    // Waits until 'pred' is true; wait_for in a loop, since condition_variable::wait needs a newer libstdc++ than MATLAB's
    template <typename Pred>
    void wait(std::unique_lock<std::mutex> &lk, Pred pred)
    {
        while (!pred())
            cv.wait_for(lk, std::chrono::milliseconds(10));
    }

    void rethrow() const
    {
        if (error == 1)
            throw std::invalid_argument(error_msg);
        if (error == 2)
            throw std::bad_alloc();
        if (error != 0)
            throw std::runtime_error(error_msg);
    }

    // Reads chunk 'k' into 'b'; only the first 'n_req' outputs (in the order of the wrapper outputs, center_freq = 0)
    void read(qd_qrt_chunk &b, arma::uword k)
    {
        const arma::uword first = k * chunk_size, last = std::min(first + chunk_size, (arma::uword)i_cir.n_elem) - 1;
        b.i_cir = i_cir.subvec(first, last);
        auto r = [this](int i, auto *p)
        { return n_req > i ? p : nullptr; };
        qd_qrt_file_read_impl<double>("", b.i_cir, i_orig, uplink, nullptr, r(1, &b.tx_pos), r(2, &b.tx_orientation),
                                      r(3, &b.rx_pos), r(4, &b.rx_orientation), r(5, &b.fbs_pos), r(6, &b.lbs_pos),
                                      r(7, &b.path_gain), r(8, &b.path_length), r(9, &b.polmat), r(10, &b.aod),
                                      r(11, &b.eod), r(12, &b.aoa), r(13, &b.eoa), r(14, &b.path_coord),
                                      normalize_polmat, r(15, &b.n_interact), r(16, &b.coord), r(17, &b.interact_type),
                                      &stream, &cache, &bytes_read, &repeat);
    }
};

// Reads the CIRs 'i_cir' (empty = all) of origin 'i_orig' in chunks of 'chunk_size' CIRs
// - init(center_freq, n_out) is called once after all indices are checked, before any CIR is read
// - consume(chunk, offset) is called on the calling thread for each chunk in order; 'offset' is the output position
//   of the first CIR of the chunk
// - 'Unlock' is a scoped type constructed while the calling thread waits for the reader (e.g. py::gil_scoped_release)
// - Errors of the reader are rethrown on the calling thread after the reader has stopped
template <typename Unlock, typename Init, typename Consume>
static inline void qd_qrt_read_chunked(const std::string &fn, const arma::uvec &i_cir, arma::uword i_orig, bool uplink,
                                       int normalize_polmat, int n_req, Init init, Consume consume, arma::uword chunk_size = 16)
{
    auto st = std::make_shared<qd_qrt_chunk_state>();
    st->stream.open(fn, std::ios::in | std::ios::binary);
    if (!st->stream.is_open())
        throw std::runtime_error("Cannot open file '" + fn + "'.");
    st->cache = quadriga_lib::qrt_read_cache_init(fn, &st->stream);

    // Call without path outputs: checks all indices and options, reads the frequencies, but no path data, and counts
    // the repeats of the whole list for the byte limit of the chunks
    arma::vec center_freq;
    qd_qrt_file_read_impl<double>("", i_cir, i_orig, uplink, &center_freq, nullptr, nullptr, nullptr, nullptr,
                                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                  nullptr, normalize_polmat, nullptr, nullptr, nullptr, &st->stream, &st->cache,
                                  &st->bytes_read, &st->repeat);

    st->i_cir = i_cir;
    if (st->i_cir.empty() && st->cache.n_cir != 0)
        st->i_cir = arma::regspace<arma::uvec>(0, (arma::uword)st->cache.n_cir - 1);
    const arma::uword n_out = st->i_cir.n_elem;
    st->i_orig = i_orig, st->uplink = uplink, st->normalize_polmat = normalize_polmat, st->n_req = n_req;
    st->chunk_size = std::max((n_req < 6) ? n_out : chunk_size, (arma::uword)1); // Positions only: one chunk, no path data
    st->n_chunk = (n_out + st->chunk_size - 1) / st->chunk_size;

    init(center_freq, n_out);
    if (n_req < 2 || n_out == 0) // Only center_freq requested or nothing to read
        return;

    if (st->n_chunk == 1) // One chunk: no thread
    {
        {
            Unlock unlock;
            st->read(st->slot[0], 0);
        }
        consume(st->slot[0], (arma::uword)0);
        return;
    }

    std::thread reader([st]()
                       {
        for (arma::uword k = 0; k < st->n_chunk; ++k)
        {
            {
                std::unique_lock<std::mutex> lk(st->mtx);
                st->wait(lk, [&] { return st->n_produced - st->n_consumed < 2 || st->stop; });
                if (st->stop)
                    return;
            }
            int err = 0;
            std::string msg;
            try
            {
                st->read(st->slot[k % 2], k);
            }
            catch (const std::invalid_argument &e)
            {
                err = 1, msg = qd_qrt_shift_prefix(e.what(), k * st->chunk_size);
            }
            catch (const std::bad_alloc &)
            {
                err = 2;
            }
            catch (const std::exception &e)
            {
                err = 3, msg = qd_qrt_shift_prefix(e.what(), k * st->chunk_size);
            }
            catch (...)
            {
                err = 3, msg = "Unknown error while reading the file.";
            }
            if (err != 0)
            {
                {
                    std::lock_guard<std::mutex> lk(st->mtx);
                    st->error = err, st->error_msg = msg;
                }
                st->cv.notify_all();
                return;
            }
            {
                std::lock_guard<std::mutex> lk(st->mtx);
                ++st->n_produced;
            }
            st->cv.notify_all();
        } });

    // Stops and joins the reader when the loop ends, also by an exception in 'consume'
    struct guard_t
    {
        std::thread &t;
        qd_qrt_chunk_state &s;
        void join()
        {
            if (!t.joinable())
                return;
            {
                std::lock_guard<std::mutex> lk(s.mtx);
                s.stop = true;
            }
            s.cv.notify_all();
            Unlock unlock;
            t.join();
        }
        ~guard_t() { join(); }
    } guard{reader, *st};

    for (arma::uword k = 0; k < st->n_chunk; ++k)
    {
        {
            Unlock unlock;
            std::unique_lock<std::mutex> lk(st->mtx);
            st->wait(lk, [&]
                     { return st->n_produced > k || st->error != 0; });
            if (st->n_produced <= k) // Reader failed before this chunk
                break;
        }
        consume(st->slot[k % 2], k * st->chunk_size);
        {
            std::lock_guard<std::mutex> lk(st->mtx);
            ++st->n_consumed;
        }
        st->cv.notify_all();
    }

    guard.join();
    st->rethrow();
}

// No-op unlock for callers without a lock to release while waiting (MEX)
struct qd_qrt_no_unlock
{
    qd_qrt_no_unlock() {}
};

#endif
