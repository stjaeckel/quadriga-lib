// Benchmark of the channel file I/O: writes all slots of a file with 'hdf5_write' and reads them back with
// 'hdf5_read_channel', reporting wall time, CPU time, throughput and file size.
//
// Two data mixes, each about 5 GB on disk:
//   a  MIMO:              3200 channels x  10 snapshots x 16x4 antennas x 200 paths
//                         (coefficients, per-antenna-pair delays, positions, orientations)
//   b  Ray-tracing route: 7000 channels x 100 snapshots x SISO x 20 paths with all path fields
//                         (coefficients, delays, path gain and length, polarization, angles, FBS/LBS positions,
//                          interaction counts and coordinates, positions, orientations)
//
// The data comes from a few pseudo-random template channels (fixed seed) that are varied per slot (name, TX
// position), so memory stays small and data generation is outside the timed regions. Only the library calls are
// timed. After writing, the file is flushed to disk (fsync, timed separately) and its page cache is dropped
// (posix_fadvise, Linux only), so the reads come from disk.
//
// Compile (add the HDF5 library only if quadriga-lib was built with it):
// g++-12 -std=c++17 -O3 -fopenmp -I/sjc/quadriga-lib/include -o benchmark_hdf5 benchmark_hdf5.cpp /sjc/quadriga-lib/lib/libquadriga.a [-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5]
//
// Usage: benchmark_hdf5 [-d <dir>] [-m a|b|both] [-s <scale>] [-k]
//   -d  output directory on a disk, not a RAM file system (default: current directory)
//   -m  mix to run (default both)
//   -s  scale factor for the number of channels, e.g. 0.01 for a smoke run (default 1)
//   -k  keep the files (default: delete them at the end)

#include "quadriga_lib.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif

using channel = quadriga_lib::channel<float>;

struct mix_def
{
    const char *id, *label;
    unsigned n_channel, n_snap, n_rx, n_tx, n_path;
    bool ray_tracing; // all path fields (true) or coefficients and delays only (false)
};

// Pseudo-random channel with the fields of the mix
static channel make_template(const mix_def &m)
{
    channel c;
    c.center_frequency = {3.5e9f};
    c.tx_pos = arma::randn<arma::fmat>(3, 1) * 100.0f;
    c.rx_pos = arma::randn<arma::fmat>(3, m.n_snap) * 100.0f;
    c.tx_orientation = arma::randu<arma::fmat>(3, 1);
    c.rx_orientation = arma::randu<arma::fmat>(3, m.n_snap);
    for (unsigned s = 0; s < m.n_snap; ++s)
    {
        c.coeff_re.push_back(arma::randn<arma::fcube>(m.n_rx, m.n_tx, m.n_path));
        c.coeff_im.push_back(arma::randn<arma::fcube>(m.n_rx, m.n_tx, m.n_path));
        c.delay.push_back(arma::randu<arma::fcube>(m.n_rx, m.n_tx, m.n_path) * 1e-6f);
        if (!m.ray_tracing)
            continue;
        c.path_gain.push_back(arma::randu<arma::fvec>(m.n_path));
        c.path_length.push_back(arma::randu<arma::fvec>(m.n_path) * 300.0f);
        c.path_polarization.push_back(arma::randn<arma::fmat>(8, m.n_path));
        c.path_angles.push_back(arma::randn<arma::fmat>(m.n_path, 4));
        c.path_fbs_pos.push_back(arma::randn<arma::fmat>(3, m.n_path) * 100.0f);
        c.path_lbs_pos.push_back(arma::randn<arma::fmat>(3, m.n_path) * 100.0f);
        c.no_interact.push_back(arma::randi<arma::Col<unsigned>>(m.n_path, arma::distr_param(0, 3)));
        c.interact_coord.push_back(arma::randn<arma::fmat>(3, arma::accu(c.no_interact.back())) * 100.0f);
    }
    return c;
}

int main(int argc, char *argv[])
{
    std::string dir = ".", which = "both";
    double scale = 1.0;
    bool keep = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "-k")
            keep = true;
        else if (i + 1 < argc && arg == "-d")
            dir = argv[++i];
        else if (i + 1 < argc && arg == "-m")
            which = argv[++i];
        else if (i + 1 < argc && arg == "-s")
            scale = std::atof(argv[++i]);
        else
        {
            std::printf("Usage: %s [-d <dir>] [-m a|b|both] [-s <scale>] [-k]\n", argv[0]);
            return 1;
        }
    }
    std::filesystem::create_directories(dir);

    const mix_def mixes[] = {{"a", "MIMO", 3200, 10, 16, 4, 200, false},
                             {"b", "ray-tracing route", 7000, 100, 1, 1, 20, true}};

    using clock = std::chrono::steady_clock;
    auto seconds = [](clock::time_point t0)
    { return std::chrono::duration<double>(clock::now() - t0).count(); };
    auto cpu_seconds = [](std::clock_t c0)
    { return double(std::clock() - c0) / CLOCKS_PER_SEC; };

    for (const auto &m : mixes)
    {
        if (which != "both" && which != m.id)
            continue;

        const unsigned n_channel = std::max(1u, unsigned(m.n_channel * scale + 0.5));
        const std::string fn = (std::filesystem::path(dir) / ("benchmark_hdf5_" + std::string(m.id) + ".hdf5")).string();
        std::filesystem::remove(fn);

        arma::arma_rng::set_seed(1);
        std::vector<channel> templates;
        for (int i = 0; i < 4; ++i)
            templates.push_back(make_template(m));

        std::printf("Mix %s (%s): %u channels x %u snapshots x %ux%u antennas x %u paths\n",
                    m.id, m.label, n_channel, m.n_snap, m.n_rx, m.n_tx, m.n_path);

        // Write: create the file, then one 'hdf5_write' per slot
        double t_write = 0.0, cpu_write = 0.0;
        {
            auto t0 = clock::now();
            std::clock_t c0 = std::clock();
            quadriga_lib::hdf5_create(fn, n_channel);
            t_write += seconds(t0), cpu_write += cpu_seconds(c0);
        }
        for (unsigned i = 0; i < n_channel; ++i)
        {
            channel &c = templates[i % templates.size()];
            c.name = "ch_" + std::to_string(i);
            c.tx_pos(0) = float(i);

            auto t0 = clock::now();
            std::clock_t c0 = std::clock();
            quadriga_lib::hdf5_write(&c, fn, i);
            t_write += seconds(t0), cpu_write += cpu_seconds(c0);
        }

        // Flush the file to disk and drop its page cache
        double t_sync = 0.0;
#ifdef __linux__
        {
            auto t0 = clock::now();
            int fd = open(fn.c_str(), O_RDONLY);
            if (fd < 0 || fsync(fd) != 0 || posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) != 0)
                std::printf("WARNING: cannot flush or drop the page cache of %s\n", fn.c_str());
            if (fd >= 0)
                close(fd);
            t_sync = seconds(t0);
        }
#else
        std::printf("WARNING: page cache not dropped (Linux only), reads may come from memory\n");
#endif

        // Read: one 'hdf5_read_channel' per slot; the content check is not timed
        double t_read = 0.0, cpu_read = 0.0;
        unsigned n_bad = 0;
        for (unsigned i = 0; i < n_channel; ++i)
        {
            auto t0 = clock::now();
            std::clock_t c0 = std::clock();
            channel c = quadriga_lib::hdf5_read_channel<float>(fn, i);
            t_read += seconds(t0), cpu_read += cpu_seconds(c0);

            const channel &ref = templates[i % templates.size()];
            if (c.name != "ch_" + std::to_string(i) || c.n_snap() != ref.n_snap() || c.tx_pos(0) != float(i) ||
                c.path_angles.size() != ref.path_angles.size() || !arma::approx_equal(c.coeff_re.back(), ref.coeff_re.back(), "absdiff", 0.0f))
                ++n_bad;
        }

        const double mb = double(std::filesystem::file_size(fn)) / 1.0e6;
        std::printf("  file size     %10.1f MB  (%.3f MB per channel)\n", mb, mb / n_channel);
        std::printf("  write         %10.2f s wall  %10.2f s CPU  %8.1f MB/s\n", t_write, cpu_write, mb / t_write);
        std::printf("  write + fsync %10.2f s wall                    %8.1f MB/s\n", t_write + t_sync, mb / (t_write + t_sync));
        std::printf("  read          %10.2f s wall  %10.2f s CPU  %8.1f MB/s\n", t_read, cpu_read, mb / t_read);
        if (n_bad != 0)
            std::printf("  ERROR: %u channels read back with wrong content\n", n_bad);
        std::printf("RESULT mix=%s channels=%u size_MB=%.1f write_s=%.3f write_cpu_s=%.3f sync_s=%.3f read_s=%.3f read_cpu_s=%.3f bad=%u\n",
                    m.id, n_channel, mb, t_write, cpu_write, t_sync, t_read, cpu_read, n_bad);
        std::fflush(stdout);

        if (!keep)
            std::filesystem::remove(fn);
    }
    return 0;
}
