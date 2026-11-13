// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Common part of the fuzz targets in tests/fuzzing/targets (see tests/fuzzing/README.md).
//
// A target file includes this header and defines two functions:
//
//     void fuzz_seeds(const fz::seeds &s);             // writes the starting inputs of the corpus
//     void fuzz_one(const uint8_t *data, size_t size); // runs the library on one input
//
// fz::input turns the fuzz input into sizes, indices and values for API calls; fz::arrayant builds an array antenna
// from it.
//
// fuzz_one passes every exception of type std::exception to fz::clean, which counts it as the correct answer to
// invalid input. Anything else that ends the input (a sanitizer report, a signal, an exception of another type, a
// timeout or too much memory) is a finding.

#pragma once

#include "quadriga_lib.hpp"

#include <fuzzer/FuzzedDataProvider.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unistd.h>

namespace fz
{
    // Writes the seed inputs into the corpus folder
    struct seeds
    {
        std::filesystem::path dir;

        // Path of a seed file; the library writes the file itself (e.g. qdant_write)
        std::string path(const std::string &name) const
        {
            return (dir / ("seed_" + name)).string();
        }

        // Seed with the given content
        void text(const std::string &name, const std::string &content) const
        {
            std::ofstream(path(name), std::ios::binary) << content;
        }

        // Copy of a file in the source tree (path relative to the repository root), at most max_bytes long;
        // a missing file is skipped, so a target keeps working when test data moves
        void copy(const std::string &repo_file, size_t max_bytes = 1 << 20) const
        {
            std::ifstream in(std::string(QD_SOURCE_DIR) + "/" + repo_file, std::ios::binary);
            if (!in)
            {
                std::cerr << "Seed file not found, skipped: " << repo_file << "\n";
                return;
            }
            std::string content(max_bytes, '\0');
            in.read(&content[0], (std::streamsize)max_bytes);
            content.resize((size_t)in.gcount());
            text(std::filesystem::path(repo_file).filename().string(), content);
        }
    };

    // Folder for the input files of this process: a subfolder per process of QD_FUZZ_TMP (set by run.py, on a RAM
    // disk where available), else of the system temp folder
    inline const std::filesystem::path &work_dir()
    {
        static const std::filesystem::path dir = []
        {
            const char *base = std::getenv("QD_FUZZ_TMP");
            std::filesystem::path d = base ? std::filesystem::path(base) : std::filesystem::temp_directory_path();
            d /= "qd_fuzz_" + std::to_string(getpid());
            std::filesystem::create_directories(d);
            return d;
        }();
        return dir;
    }

    // Write an input to a file in the work folder; readers that need an extension get it through the name
    inline std::string write_file(const std::string &name, const uint8_t *data, size_t size)
    {
        std::string fn = (work_dir() / name).string();
        std::ofstream(fn, std::ios::binary | std::ios::trunc).write((const char *)data, (std::streamsize)size);
        return fn;
    }
    inline std::string write_file(const std::string &name, const std::string &content)
    {
        return write_file(name, (const uint8_t *)content.data(), content.size());
    }

    // Run f; a std::exception is the clean answer to invalid input
    template <typename F>
    inline void clean(F &&f)
    {
        try
        {
            f();
        }
        catch (const std::exception &)
        {
        }
    }

    // Structured input for targets that call API functions: sizes, indices and values taken from the fuzz input
    struct input : FuzzedDataProvider
    {
        using FuzzedDataProvider::FuzzedDataProvider;

        // A size in [0, max]
        arma::uword size(arma::uword max)
        {
            return ConsumeIntegralInRange<arma::uword>(0, max);
        }

        // An index for a container of n elements: valid, or just past the end, or the largest value (-1)
        arma::uword index(arma::uword n)
        {
            switch (ConsumeIntegralInRange<int>(0, 7))
            {
            case 0:
                return n;
            case 1:
                return n + 1;
            case 2:
                return std::numeric_limits<arma::uword>::max();
            default:
                return n == 0 ? 0 : ConsumeIntegralInRange<arma::uword>(0, n - 1);
            }
        }

        // A value: mostly moderate numbers, sometimes a special value (zero, signed zero, NaN, Inf, extremes)
        template <typename T>
        T value()
        {
            const T special[] = {T(0), -T(0), T(1), T(-1), std::numeric_limits<T>::quiet_NaN(),
                                 std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(),
                                 std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest(),
                                 std::numeric_limits<T>::denorm_min(), std::numeric_limits<T>::epsilon()};
            uint8_t sel = ConsumeIntegral<uint8_t>();
            if (sel < 32)
                return special[sel % (sizeof(special) / sizeof(T))];
            if (sel < 224)
                return ConsumeFloatingPointInRange<T>(T(-100), T(100));
            return ConsumeFloatingPoint<T>();
        }

        template <typename T>
        arma::Col<T> col(arma::uword n)
        {
            arma::Col<T> x(n);
            for (auto &v : x)
                v = value<T>();
            return x;
        }

        template <typename T>
        arma::Mat<T> mat(arma::uword n_rows, arma::uword n_cols)
        {
            arma::Mat<T> x(n_rows, n_cols);
            for (auto &v : x)
                v = value<T>();
            return x;
        }

        template <typename T>
        arma::Cube<T> cube(arma::uword n_rows, arma::uword n_cols, arma::uword n_slices)
        {
            arma::Cube<T> x(n_rows, n_cols, n_slices);
            for (auto &v : x)
                v = value<T>();
            return x;
        }
    };

    // An arrayant with up to 8 elevation, 12 azimuth samples and 6 elements; sizes mismatch in about one input of four
    template <typename T>
    inline quadriga_lib::arrayant<T> arrayant(input &in)
    {
        quadriga_lib::arrayant<T> a;
        const bool consistent = in.ConsumeIntegralInRange<int>(0, 3) != 0;
        arma::uword n_el = in.size(8), n_az = in.size(12), n_e = in.size(6), n_p = in.size(6);
        auto sz = [&](arma::uword n) { return consistent ? n : in.size(n + 1); };

        // Regular grid, sorted whole degrees (at least 1 degree apart where not equal), or any values
        auto grid = [&](arma::uword n, int lim_deg)
        {
            const T deg = T(3.14159265358979 / 180.0);
            arma::Col<T> g(n);
            switch (in.ConsumeIntegralInRange<int>(0, 2))
            {
            case 0:
                return arma::Col<T>(arma::linspace<arma::Col<T>>(-T(lim_deg) * deg, T(lim_deg) * deg, n));
            case 1:
                for (auto &v : g)
                    v = T(in.ConsumeIntegralInRange<int>(-lim_deg, lim_deg)) * deg;
                return arma::Col<T>(arma::sort(g));
            default:
                return in.col<T>(n);
            }
        };
        a.azimuth_grid = grid(sz(n_az), 180);
        a.elevation_grid = grid(sz(n_el), 90);

        a.e_theta_re = in.cube<T>(n_el, n_az, n_e);
        a.e_theta_im = in.cube<T>(sz(n_el), sz(n_az), sz(n_e));
        a.e_phi_re = in.cube<T>(sz(n_el), sz(n_az), sz(n_e));
        a.e_phi_im = in.cube<T>(sz(n_el), sz(n_az), sz(n_e));

        switch (in.ConsumeIntegralInRange<int>(0, 2))
        {
        case 1:
            a.element_pos = in.mat<T>(3, n_e);
            break;
        case 2:
            a.element_pos = in.mat<T>(in.size(4), in.size(n_e + 1));
            break;
        }
        switch (in.ConsumeIntegralInRange<int>(0, 3))
        {
        case 1:
            a.coupling_re = in.mat<T>(n_e, n_p), a.coupling_im = in.mat<T>(n_e, n_p);
            break;
        case 2:
            a.coupling_re = in.mat<T>(n_e, n_p);
            break;
        case 3:
            a.coupling_re = in.mat<T>(in.size(n_e + 1), in.size(n_p + 1));
            a.coupling_im = in.mat<T>(in.size(n_e + 1), in.size(n_p + 1));
            break;
        }
        a.center_frequency = in.value<T>();
        a.read_only = in.ConsumeIntegralInRange<int>(0, 7) == 0;

        // Validated, then grid values changed in place (keeps the data pointers of the quick check)
        if (in.ConsumeBool())
        {
            clean([&] { a.is_valid(); });
            if (in.ConsumeBool() && a.azimuth_grid.n_elem != 0)
                a.azimuth_grid(in.ConsumeIntegralInRange<arma::uword>(0, a.azimuth_grid.n_elem - 1)) = in.value<T>();
            if (in.ConsumeBool() && a.elevation_grid.n_elem != 0)
                a.elevation_grid(in.ConsumeIntegralInRange<arma::uword>(0, a.elevation_grid.n_elem - 1)) = in.value<T>();
        }
        return a;
    }
}

void fuzz_seeds(const fz::seeds &s);
void fuzz_one(const uint8_t *data, size_t size);

// "--write-seeds=<folder>" writes the seeds and exits; libFuzzer itself ignores arguments that start with "--"
extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv)
{
    const std::string flag = "--write-seeds=";
    for (int i = 1; i < *argc; ++i)
        if (std::strncmp((*argv)[i], flag.c_str(), flag.size()) == 0)
        {
            fz::seeds s{std::filesystem::path((*argv)[i] + flag.size())};
            try
            {
                std::filesystem::create_directories(s.dir);
                fuzz_seeds(s);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Writing the seeds failed: " << e.what() << "\n";
                std::exit(1);
            }
            std::exit(0);
        }
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    fuzz_one(data, size);
    return 0;
}
