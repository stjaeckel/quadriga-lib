// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Layout CSV files: layout_csv_read with the fuzz input as the Tx file, as the Rx file and as both, alone and together
// with a valid file of the other side (so that the pairing, the shared antenna list and the band rules of two files are
// reached), without and with a given frequency list. A layout that reads must be valid; it is written (compact and with
// every cell) and read again, and all members that the files hold must be the same (angles within their 15 digits).
// The antennas are not evaluated: arbitrary panel sizes would make single runs slow or exhaust the memory. Every node of a
// layout that reads is also sampled with sample_track, with a fixed list of times and without times (offsets, sizes and the
// row times of tracks of any shape).

#include "fuzz_target.hpp"
#include "quadriga_layout.hpp"
#include "../../catch2_tests/qd_test_layout.hpp" // qd_layout_diff: the members that the files hold

#include <cstdlib>
#include <string>

static const char *valid_tx = "name,sector,x,y,z,heading,tilt,center_freq,antenna,panel,seed,layout_seed,wrap,d_clutter\n"
                              "BS1,0,0,0,25,30,10,3.5e9,3gpp,\"4,4,1,2\",5,7,\"100,0;0,200\",3\n"
                              "BS1,0,,,,,,28e9,3gpp,\"8,8,1,4\"\n"
                              "BS1,1,,,,150\n"
                              "BS2,0,500,0,25,90,0,3.5e9,dipole\n"
                              "BS2,,,,,,,28e9\n";
static const char *valid_rx = "name,x,y,z,heading,speed,direction_az,scenario,antenna,pairing\n"
                              "UT1,10,20,1.5,45,0.8,120,UMa,omni,\"BS1,BS2\"\n"
                              "UT2,0,0,1.5,0,,,,,BS2\n"
                              "UT2,1,0,1.5,90,,,UMa_NLOS\n";

void fuzz_seeds(const fz::seeds &s)
{
    s.text("valid_tx.csv", valid_tx);
    s.text("valid_rx.csv", valid_rx);
    s.text("pairing.csv", "name,x,y,z,pairing,relative,seed\n\"a,b\",0,0,0,,,9007199254740993\nU,1,1,1,\"a,b\",1,\nV,0,0,0,,,\n");
    s.text("qdant.csv", "name,x,y,z,antenna,qdant_file,qdant_id\nBS,0,0,25,qdant,../ants.qdant,2\nBS2,1,0,25,qdant,ants.qdant,\n");
    s.text("time.csv", "name,x,y,z,time,indoor,d2d_in,floor,relative,scenpar.DS_VS\nA,0,0,0,0,1,12,2,1,0.2\nA,1,0,0,2.5,0,NaN,1,,\nB,5,5,5,,,,,,\n");
    s.text("misc.csv", "\xEF\xBB\xBFname,sector,x,y,z,foo\r\n# c\r\n\"A, \"\"1\"\"\",0,1,2,3,x\r\n\"A, \"\"1\"\"\",1,,,,\r\nB,0,1,1,1\r\n");
}

static void sample_all(const quadriga_lib::layout_nodes &N)
{
    const arma::vec times = {-1.0, 0.0, 0.5, 2.0, 1.0e3};
    for (arma::uword i = 0; i < N.n_node(); ++i)
        for (const arma::vec &t : {arma::vec(), times})
        {
            arma::mat pos, ori;
            arma::uvec row;
            bool ok = false;
            fz::clean([&]
                      {
                quadriga_lib::sample_track(N, i, t, pos, ori, &row);
                ok = true; });
            if (!ok)
                continue; // A track without time and speed cannot be sampled at times
            const arma::uword n = t.is_empty() ? N.n_row(i) : t.n_elem;
            if (pos.n_cols != n || ori.n_cols != n || row.n_elem != n || pos.n_rows != 3 || ori.n_rows != 3)
                std::abort(); // Wrong output size
            for (arma::uword k = 0; k < n; ++k)
                if (row[k] < N.offset[i] || row[k] >= N.offset[i] + N.n_row(i))
                    std::abort(); // The row is not a row of the node
        }
}

static void check(const std::string &fn_tx, const std::string &fn_rx, const arma::vec &freq)
{
    quadriga_lib::layout L;
    bool read = false;
    fz::clean([&]
              {
        std::vector<std::string> extra_tx, extra_rx;
        L = quadriga_lib::layout_csv_read(fn_tx, fn_rx, freq, false, &extra_tx, &extra_rx);
        read = true; });
    if (!read)
        return;
    if (!L.is_valid().empty())
        std::abort(); // The reader returned a layout that is not valid
    sample_all(L.tx);
    sample_all(L.rx);

    const std::string tx = fz::work_dir().string() + "/out_tx.csv", rx = fz::work_dir().string() + "/out_rx.csv";
    for (bool compact : {true, false})
    {
        bool written = false;
        fz::clean([&]
                  {
            quadriga_lib::layout_csv_write(tx, rx, L, compact);
            written = true; });
        if (!written)
            continue; // Some layouts that read cannot be written (e.g. a scenario that returns to none)
        quadriga_lib::layout M;
        try
        {
            M = quadriga_lib::layout_csv_read(tx, rx, {}, false);
        }
        catch (const std::exception &)
        {
            std::abort(); // A written layout must read
        }
        if (!qd_layout_diff(L, M).empty())
            std::abort(); // Everything the files hold must come back (angles within the digits of the file)
    }
}

void fuzz_one(const uint8_t *data, size_t size)
{
    const std::string fuzz = fz::write_file("fuzz.csv", data, size), fuzz2 = fz::write_file("fuzz2.csv", data, size);
    const std::string ok_tx = fz::write_file("ok_tx.csv", std::string(valid_tx)), ok_rx = fz::write_file("ok_rx.csv", std::string(valid_rx));
    const arma::vec freq = {3.5e9, 28e9};

    check(fuzz, "", {});
    check("", fuzz, {});
    check(fuzz, ok_rx, {});
    check(ok_tx, fuzz, {});
    check(fuzz, fuzz2, {});
    check(fuzz, ok_rx, freq);
    check(ok_tx, fuzz, freq);
}
