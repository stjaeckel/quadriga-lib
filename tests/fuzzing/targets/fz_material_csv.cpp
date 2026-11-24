// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Material tables (CSV) for the EM and acoustic material models: obj_file_read of a fixed OBJ file with the fuzz
// input as the material table, in the lenient and in the strict mode; every material of a table that reads is then
// evaluated by medium_gain, interface_gain and refractive_index, in EM and in scalar mode. The same input goes to the
// CSV unit itself (qd_csv.hpp, the reader of the material table): parsed with each combination of its options, every
// cell read as text, number, list and matrix, written out and read again (the result must be the same table)

#include "fuzz_target.hpp"
#include "qd_csv.hpp"

#include <sstream>

// Faces with materials of the RF table, the acoustic table and one that is in neither
static const char *fixed_obj = "o Room\n"
                               "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
                               "usemtl itu_concrete\nf 1 2 3\n"
                               "usemtl concrete_150mm\nf 1 2 4\n"
                               "usemtl air\nf 1 3 4\n"
                               "usemtl unknown_material\nf 2 3 4\n";

void fuzz_seeds(const fz::seeds &s)
{
    s.copy("data/materials_rf.csv");
    s.copy("data/materials_acoustic.csv");

    // Quoted cells with commas, semicolons and escaped quotes, comments, blank lines, a byte order mark, CRLF, short
    // and long rows, a last line without line end, list and matrix cells, NaN and Inf
    s.text("quotes.csv", "name,a,b,c,note\n\"itu_concrete, thick\",5.24,0.0,0.0462,\"say \"\"hi\"\"; ok\"\n"
                         "\"air\" ,1,\"0\",,\n");
    s.text("comments.csv", "# material table\n\nname,a,b\n  # indented comment\nair,1,0\n\nwall,5.24,0.5\n");
    s.text("bom_crlf.csv", "\xEF\xBB\xBFname,a,b,c\r\nair,1,0,0\r\nwall,5.24\r\n\"w2\",1,2,3,4,5\r\nlast,1,2,3");
    s.text("cells.csv", "name,a,b,c\nr1,1e3,-.5,+2.\nr2,\"1,2,3\",\"6,0;10,0\",nan\nitu_wood,inf,-Inf,1e999\nr4,0x10,1e,--1\n");
}

// The CSV unit: all option combinations, every cell as text and as number, list and matrix, then a round trip
static void fuzz_csv(const uint8_t *data, size_t size)
{
    const std::string text((const char *)data, size);
    for (int k = 0; k < 4; ++k)
    {
        qd_csv::options opt;
        opt.allow_long_rows = (k & 1) != 0;
        opt.allow_duplicate_header = (k & 2) != 0;
        fz::clean([&]
                  {
            const qd_csv::table T = qd_csv::parse(text, "fuzz.csv", opt);
            qd_csv::table copy;
            copy.header = T.header;
            for (size_t r = 0; r < T.n_rows(); ++r)
            {
                std::vector<std::string_view> row;
                for (size_t c = 0; c < T.n_cols(); ++c)
                {
                    row.push_back(T.cell(r, c));
                    fz::clean([&] { T.get_double(r, c, 0.0, (r + c) % 2 == 0, (r + c) % 3 != 0); });
                    fz::clean([&] { T.get_list(r, c, (r + c) % 2 == 0, (r + c) % 3 != 0); });
                    fz::clean([&] { T.get_matrix(r, c, (r + c) % 2 == 0, (r + c) % 3 != 0); });
                }
                copy.add_row(row);
                T.line(r);
            }
            for (const std::string &name : T.header)
                T.find(name);

            // Written out and read again: whatever the writer accepts must read back as the same header and cells
            std::string out;
            try
            {
                std::ostringstream os;
                qd_csv::write(os, copy);
                out = os.str();
            }
            catch (const std::invalid_argument &)
            {
                return; // a table that cannot be written is a clean error
            }
            const qd_csv::table back = qd_csv::parse(out, "fuzz_out.csv", opt);
            bool same = back.header == copy.header && back.n_rows() == copy.n_rows();
            for (size_t r = 0; same && r < copy.n_rows(); ++r)
                for (size_t c = 0; same && c < copy.n_cols(); ++c)
                    same = back.cell(r, c) == copy.cell(r, c);
            if (!same)
            {
                std::fprintf(stderr, "qd_csv: write and parse do not give back the table\n");
                std::abort();
            }
            });
    }
}

void fuzz_one(const uint8_t *data, size_t size)
{
    fuzz_csv(data, size);

    std::string fn_csv = fz::write_file("in.csv", data, size);
    std::string fn = fz::write_file("in.obj", fixed_obj);

    for (bool strict : {false, true})
        fz::clean([&]
                  {
            arma::mat mesh;
            arma::uvec csv_ind;
            std::vector<std::string> csv_names;
            std::unordered_map<std::string, std::vector<double>> csv_prop;
            quadriga_lib::obj_file_read<double>(fn, &mesh, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                nullptr, fn_csv, &csv_ind, &csv_names, &csv_prop, strict);
            for (arma::uword iM = 1; iM <= (arma::uword)csv_names.size(); ++iM)
                for (bool scalar_mode : {false, true})
                {
                    const double f = scalar_mode ? 1000.0 : 3.5e9; // 1 kHz acoustic, 3.5 GHz EM
                    quadriga_lib::medium_gain<double>(csv_prop, iM, 0.1, f, scalar_mode);
                    quadriga_lib::interface_gain<double>(csv_prop, iM, f, scalar_mode);
                    quadriga_lib::refractive_index<double>(csv_prop, iM, f, scalar_mode);
                } });
}
