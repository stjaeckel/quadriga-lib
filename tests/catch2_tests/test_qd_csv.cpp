// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// The private CSV unit qd_csv.hpp: quoted cells and escapes, comments and blank lines, byte order mark, CRLF, short and
// long rows, header rules, number syntax and conversion, list and matrix cells, line numbers in the errors, the writer
// (quoting rules, number format, round trips), the independence of the global locale, a table of 100,000 rows, and the
// material reader of obj_file_read, which is built on the unit.
//
// Oracles: texts and values typed by hand. The expected number texts (the fewest of 15, 16, 17 significant digits that
// read back) were produced with Python's '%.{15,16,17}g' and float(); the expected doubles are C++ literals, which the
// compiler converts. The round trips use the reader as the supporting function.

#include <catch2/catch_test_macros.hpp>

#include "qd_csv.hpp"
#include "quadriga_tools.hpp"

#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <locale>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
using qd_csv::number_status;

qd_csv::table parse(const std::string &text, const qd_csv::options &opt = qd_csv::options())
{
    return qd_csv::parse(text, "mem.csv", opt);
}

// Message of the exception a call throws, or "" if it does not throw
template <class F>
std::string what(F f)
{
    try
    {
        f();
    }
    catch (const std::exception &e)
    {
        return e.what();
    }
    return "";
}

std::string written(const qd_csv::table &T)
{
    std::ostringstream os;
    qd_csv::write(os, T);
    return os.str();
}

// File that is removed when it goes out of scope
struct temp_file
{
    std::string fn;
    explicit temp_file(const std::string &name, const std::string &content = "") : fn(name)
    {
        std::ofstream f(fn, std::ios::binary);
        f << content;
    }
    ~temp_file()
    {
        std::error_code ec;
        std::filesystem::remove(fn, ec);
    }
};

double number(const std::string &s, number_status *st = nullptr, bool allow_nan = false)
{
    double v = -777.0;
    const number_status r = qd_csv::parse_number(s, v, allow_nan);
    if (st != nullptr)
        *st = r;
    return v;
}

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }
} // namespace

TEST_CASE("qd_csv - quoted cells and escapes")
{
    const auto T = parse("name,value,note\n"
                         "plain,1,hello\n"
                         "\"a,b\",\"say \"\"hi\"\"\",  \"  spaced  \"  \n"
                         "\"\",\"x;y\",ab\"c\n"
                         " x , \"y\" ,z \n");
    REQUIRE(T.header == std::vector<std::string>{"name", "value", "note"});
    REQUIRE(T.n_rows() == 4);
    REQUIRE(T.n_cols() == 3);

    CHECK(T.cell(0, 0) == "plain");
    CHECK(T.cell(0, 1) == "1");
    CHECK(T.cell(0, 2) == "hello");

    CHECK(T.cell(1, 0) == "a,b");                // a quoted cell may contain a comma
    CHECK(T.cell(1, 1) == "say \"hi\"");         // '""' is a quote
    CHECK(T.cell(1, 2) == "  spaced  ");         // blanks inside the quotes are kept, those outside are not

    CHECK(T.cell(2, 0) == "");                   // a quoted empty cell
    CHECK(T.cell(2, 1) == "x;y");                // semicolon
    CHECK(T.cell(2, 2) == "ab\"c");              // a quote inside an unquoted cell is an ordinary character

    CHECK(T.cell(3, 0) == "x");                  // blanks around unquoted and quoted cells are removed
    CHECK(T.cell(3, 1) == "y");
    CHECK(T.cell(3, 2) == "z");

    // Quoted header names, and a quote doubled at the end of a cell
    const auto H = parse("\"a,1\",\"b\"\"\"\n\"q\"\"\",2\n");
    REQUIRE(H.n_cols() == 2);
    CHECK(H.header[0] == "a,1");
    CHECK(H.header[1] == "b\"");
    CHECK(H.cell(0, 0) == "q\"");
    CHECK(H.find("a,1") == 0);
    CHECK(H.find("b\"") == 1);

    // Non-ASCII text passes through unchanged
    const auto U = parse("n,t\nm,\xC2\xB5\xE2\x82\xAC\n");
    CHECK(U.cell(0, 1) == "\xC2\xB5\xE2\x82\xAC");
}

TEST_CASE("qd_csv - comments, blank lines and line numbers")
{
    const auto T = parse("# first comment\n"   // 1
                         "\n"                   // 2
                         "   # indented\n"      // 3
                         "a,b\n"                // 4: the header
                         "\n"                   // 5
                         "1,2\n"                // 6
                         "   \t\n"              // 7
                         "# between rows\n"     // 8
                         "x#y,#z\n"             // 9: '#' inside a row is no comment
                         "\"#q\",3\n"           // 10: a quoted cell that starts with '#' is a row
                         "  # last\n");         // 11
    REQUIRE(T.header == std::vector<std::string>{"a", "b"});
    REQUIRE(T.n_rows() == 3);
    CHECK(T.line(0) == 6);
    CHECK(T.line(1) == 9);
    CHECK(T.line(2) == 10);
    CHECK(T.cell(1, 0) == "x#y");
    CHECK(T.cell(1, 1) == "#z");
    CHECK(T.cell(2, 0) == "#q");

    // Header only, with and without a final line end
    CHECK(parse("a,b\n").n_rows() == 0);
    CHECK(parse("a,b").n_rows() == 0);
    CHECK(parse("# c\n\na,b").n_cols() == 2);
}

TEST_CASE("qd_csv - byte order mark, CRLF and last line without line end")
{
    const auto T = parse("\xEF\xBB\xBF" "a,b\r\n1,2\r\n\r\n# c\r\n3,4");
    REQUIRE(T.header == std::vector<std::string>{"a", "b"}); // not "\xEF\xBB\xBF" "a"
    REQUIRE(T.n_rows() == 2);
    CHECK(T.cell(0, 1) == "2");                              // no CR at the end of the last cell
    CHECK(T.cell(1, 0) == "3");
    CHECK(T.cell(1, 1) == "4");
    CHECK(T.line(0) == 2);
    CHECK(T.line(1) == 5);

    // Mixed line ends, a quoted last cell before CRLF, a BOM before a comment line
    const auto M = parse("\xEF\xBB\xBF# comment\nname,v\r\n\"x\",\"y\"\r\nz,w\n");
    REQUIRE(M.n_rows() == 2);
    CHECK(M.cell(0, 0) == "x");
    CHECK(M.cell(0, 1) == "y");
    CHECK(M.cell(1, 1) == "w");

    // A BOM that is not at the start is text
    const auto B = parse("a,b\n\xEF\xBB\xBF" "1,2\n");
    CHECK(B.cell(0, 0) == "\xEF\xBB\xBF" "1");

    // A quoted header name that starts with the bytes of a BOM is written quoted, so that it reads back
    const auto Q = parse("\"\xEF\xBB\xBF" "a\",b\n1,2\n");
    CHECK(Q.header[0] == "\xEF\xBB\xBF" "a");
    const std::string text = written(Q);
    CHECK(text == "\"\xEF\xBB\xBF" "a\",b\n1,2\n");
    CHECK(parse(text).header == Q.header);
}

TEST_CASE("qd_csv - short rows, long rows and header rules")
{
    // A short row is padded with empty cells; a trailing comma is an empty cell
    const auto S = parse("a,b,c\n1\n1,2\n1,2,\n,,\n");
    REQUIRE(S.n_rows() == 4);
    CHECK(S.cell(0, 0) == "1");
    CHECK(S.cell(0, 1) == "");
    CHECK(S.cell(0, 2) == "");
    CHECK(S.cell(1, 1) == "2");
    CHECK(S.cell(1, 2) == "");
    CHECK(S.cell(2, 2) == "");
    CHECK(S.cell(3, 0) == "");

    // A long row throws, unless the option drops the extra cells
    CHECK(what([] { parse("a,b\n1,2\n1,2,3\n"); }) == "mem.csv, line 3: the row has 3 cells, the header has 2 columns.");
    CHECK(what([] { parse("a,b\n1,2,\n"); }) == "mem.csv, line 2: the row has 3 cells, the header has 2 columns.");
    qd_csv::options opt;
    opt.allow_long_rows = true;
    const auto L = parse("a,b\n1,2,3,\"4\"\n5,6\n", opt);
    REQUIRE(L.n_rows() == 2);
    CHECK(L.cell(0, 0) == "1");
    CHECK(L.cell(0, 1) == "2");
    CHECK(L.cell(1, 0) == "5");
    CHECK(L.cell(1, 1) == "6");

    // Names are case-sensitive and trimmed; equal names throw, unless the option allows them
    const auto H = parse(" Name , name ,\"x y\"\n");
    CHECK(H.header == std::vector<std::string>{"Name", "name", "x y"});
    CHECK(H.find("name") == 1);
    CHECK(H.find("Name") == 0);
    CHECK(H.find("NAME") == qd_csv::table::npos);
    CHECK(H.find("x y") == 2);
    CHECK(H.find(" name ") == qd_csv::table::npos);

    CHECK(what([] { parse("# c\na,b,a\n1,2,3\n"); }) == "mem.csv, line 2: the column name 'a' is used twice in the header.");
    CHECK(what([] { parse("a,,\n"); }) == "mem.csv, line 1: the column name '' is used twice in the header.");
    qd_csv::options dup;
    dup.allow_duplicate_header = true;
    const auto D = parse("a,b,a\n1,2,3\n", dup);
    CHECK(D.find("a") == 0); // the first
    CHECK(D.cell(0, 2) == "3");
    CHECK(D.n_cols() == 3);
}

TEST_CASE("qd_csv - numbers")
{
    number_status st;

    // Valid numbers and their doubles (C++ literals as the oracle)
    struct ok_case
    {
        const char *text;
        double value;
    };
    const ok_case ok[] = {{"0", 0.0},
                          {"1", 1.0},
                          {"+1", 1.0},
                          {"-1", -1.0},
                          {"0.1", 0.1},
                          {".5", 0.5},
                          {"5.", 5.0},
                          {"-.25", -0.25},
                          {"+.25e1", 2.5},
                          {"1e3", 1000.0},
                          {"1E3", 1000.0},
                          {"1e+3", 1000.0},
                          {"1e-3", 1e-3},
                          {"12.5e-2", 0.125},
                          {"007", 7.0},
                          {"0.30000000000000004", 0.30000000000000004},
                          {"123456789012345678", 123456789012345678.0},
                          {"1e23", 1e23},
                          {"2.2250738585072014e-308", 2.2250738585072014e-308},
                          {"1e-320", 1e-320},                                // subnormal
                          {"4.9406564584124654e-324", 4.9406564584124654e-324}, // the smallest subnormal
                          {"1.7976931348623157e308", 1.7976931348623157e308},
                          {"1.7976931348623158e308", 1.7976931348623157e308}, // below the halfway point to 2^1024
                          {"0.1111111111111111111111111111111111111111", 0.1111111111111111111111111111111111111111},
                          {"1.0000000000000000000000000000000000000000000000000000000000000000001", 1.0}};
    for (const ok_case &c : ok)
    {
        INFO("text '" << c.text << "'");
        const double v = number(c.text, &st);
        CHECK(st == number_status::ok);
        CHECK(same_bits(v, c.value));
    }

    // Many leading zeros before the first digit, with and without a point
    CHECK(number(std::string(300, '0') + "1", &st) == 1.0);
    CHECK(number("0." + std::string(300, '0') + "1", &st) == 1e-301);
    CHECK(st == number_status::ok);

    // Zero keeps its sign; a number below the smallest subnormal is a zero with its sign
    CHECK(same_bits(number("-0"), -0.0));
    CHECK(same_bits(number("0e999999999999"), 0.0));
    CHECK(same_bits(number("-0.0e5"), -0.0));
    CHECK(same_bits(number("1e-400", &st), 0.0));
    CHECK(st == number_status::ok);
    CHECK(same_bits(number("-1e-400"), -0.0));
    CHECK(same_bits(number("1e-99999999999999999999"), 0.0));
    CHECK(same_bits(number("0." + std::string(400, '0') + "1"), 0.0));
    CHECK(same_bits(number("-0." + std::string(400, '0') + "1"), -0.0));

    // Out of range
    for (const char *text : {"1e309", "-1e309", "1e400", "1.7976931348623159e308", "1e99999999999999999999",
                             "9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999"})
    {
        INFO("text '" << text << "'");
        number(text, &st);
        CHECK(st == number_status::range);
    }

    // Inf and NaN in any case; NaN only when it is allowed
    for (const char *text : {"inf", "Inf", "INF", "iNf", "+inf", "+Inf"})
    {
        INFO("text '" << text << "'");
        CHECK(number(text, &st) == INFINITY);
        CHECK(st == number_status::ok);
    }
    for (const char *text : {"-inf", "-Inf", "-INF"})
    {
        INFO("text '" << text << "'");
        CHECK(number(text, &st) == -INFINITY);
        CHECK(st == number_status::ok);
    }
    for (const char *text : {"nan", "NaN", "NAN", "nAn", "+nan", "-NaN"})
    {
        INFO("text '" << text << "'");
        CHECK(std::isnan(number(text, &st, true)));
        CHECK(st == number_status::ok);
        number(text, &st, false);
        CHECK(st == number_status::invalid);
    }

    // Not numbers: the whole text must be consumed
    for (const char *text : {"", " ", "+", "-", ".", "+.", "e5", ".e5", "1e", "1e+", "1e-", "1e5.5", "1.5.2", "1..5", "--1",
                             "+-1", "1-", "1 2", " 1", "1 ", "1,5", "1;5", "1_0", "1x", "x1", "0x10", "0x", "0X1p3", "1f", "1d3",
                             "infinity", "in", "infx", "nan(1)", "n", "1e5e5", "\xC2\xB5", "1\xC2\xA0", "٣", "1e\xC2\xB2"})
    {
        INFO("text '" << text << "'");
        number(text, &st, true);
        CHECK(st == number_status::invalid);
        number(text, &st, false);
        CHECK(st == number_status::invalid);
    }

    // The value is not written for a result that is not 'ok'
    double v = 4.25;
    CHECK(qd_csv::parse_number("1e999", v) == number_status::range);
    CHECK(qd_csv::parse_number("x", v) == number_status::invalid);
    CHECK(v == 4.25);

    // A cell text with an embedded zero byte is not a number
    CHECK(qd_csv::parse_number(std::string_view("1\0" "2", 3), v) == number_status::invalid);
}

TEST_CASE("qd_csv - get_double")
{
    const auto T = parse("a,b,c\n"
                         " 2.5 ,,nan\n"
                         "1e999,inf,-Inf\n"
                         "0x10,1.5x,\"3 \"\n");
    CHECK(T.get_double(0, 0) == 2.5);
    CHECK(T.get_double(0, 1) == 0.0);          // an empty cell gives the default
    CHECK(T.get_double(0, 1, -4.5) == -4.5);
    CHECK(std::isnan(T.get_double(0, 1, NAN))); // also NaN
    CHECK(std::isnan(T.get_double(0, 2)));
    CHECK(T.get_double(1, 1) == INFINITY);
    CHECK(T.get_double(1, 2) == -INFINITY);
    CHECK(what([&] { T.get_double(2, 2); }) == "mem.csv, line 4, column 'c' is not numeric ('3 ')."); // quoted blanks are kept

    // NaN and Inf can be refused
    CHECK(what([&] { T.get_double(0, 2, 0.0, false, true); }) == "mem.csv, line 2, column 'c' is not numeric ('nan').");
    CHECK(what([&] { T.get_double(1, 1, 0.0, true, false); }) == "mem.csv, line 3, column 'b' is not numeric ('inf').");
    CHECK(T.get_double(1, 1, 0.0, false, true) == INFINITY);

    // The messages name file, line and column
    CHECK(what([&] { T.get_double(1, 0); }) == "mem.csv, line 3, column 'a' is out of range ('1e999').");
    CHECK(what([&] { T.get_double(2, 0); }) == "mem.csv, line 4, column 'a' is not numeric ('0x10').");
    CHECK(what([&] { T.get_double(2, 1); }) == "mem.csv, line 4, column 'b' is not numeric ('1.5x').");
}

TEST_CASE("qd_csv - list and matrix cells")
{
    const auto T = parse("v\n"
                         "\"1, 2.5 ,-3\"\n"                   // 0
                         "\n"                                  // (blank line, no row)
                         "\"\"\n"                              // 1: empty
                         "\"4\"\n"                             // 2: one entry
                         "\"1,,2\"\n"                          // 3
                         "\"1,2,\"\n"                          // 4
                         "\"1;2\"\n"                           // 5
                         "\"nan,inf,-Inf\"\n"                  // 6
                         "\"1,x\"\n"                           // 7
                         "\"1e999\"\n"                         // 8
                         "\"1,2,3;4,5,6\"\n"                   // 9: [2, 3]
                         "\"6,0\"\n"                           // 10: one row
                         "\"1;2;3\"\n"                         // 11: one column
                         "\" 6 , 0 ; 10 , 0 \"\n"              // 12: blanks
                         "\"1,2;3\"\n"                         // 13: rows of different length
                         "\"1,2;\"\n"                          // 14: empty last row
                         "\"1,2;;3,4\"\n"                      // 15: empty row
                         "\"1,;3,4\"\n"                        // 16: empty entry
                         "\"1,2;x,4\"\n"                       // 17
                         "\"nan,2;3,Inf\"\n"                   // 18
                         "\"2,2;1,1,1\"\n"                     // 19
                         "\" \"\n");                            // 20: blank only
    REQUIRE(T.n_rows() == 21);

    // Lists
    CHECK(T.get_list(0, 0) == std::vector<double>{1.0, 2.5, -3.0});
    CHECK(T.get_list(1, 0).empty());
    CHECK(T.get_list(20, 0).empty());
    CHECK(T.get_list(2, 0) == std::vector<double>{4.0});
    CHECK(what([&] { T.get_list(3, 0); }) == "mem.csv, line 6, column 'v' has an empty entry (entry 2).");
    CHECK(what([&] { T.get_list(4, 0); }) == "mem.csv, line 7, column 'v' has an empty entry (entry 3).");
    CHECK(what([&] { T.get_list(5, 0); }) == "mem.csv, line 8, column 'v' has an entry that is not numeric (entry 1 ('1;2')).");
    const auto N = T.get_list(6, 0);
    REQUIRE(N.size() == 3);
    CHECK(std::isnan(N[0]));
    CHECK(N[1] == INFINITY);
    CHECK(N[2] == -INFINITY);
    CHECK(what([&] { T.get_list(6, 0, false, true); }).find("entry 1 ('nan')") != std::string::npos);
    CHECK(what([&] { T.get_list(6, 0, true, false); }).find("entry 2 ('inf')") != std::string::npos);
    CHECK(what([&] { T.get_list(7, 0); }) == "mem.csv, line 10, column 'v' has an entry that is not numeric (entry 2 ('x')).");
    CHECK(what([&] { T.get_list(8, 0); }) == "mem.csv, line 11, column 'v' has an entry that is out of range (entry 1 ('1e999')).");

    // Matrices: cell rows are matrix rows
    const arma::mat A = T.get_matrix(9, 0);
    REQUIRE(A.n_rows == 2);
    REQUIRE(A.n_cols == 3);
    CHECK(A(0, 0) == 1.0);
    CHECK(A(0, 2) == 3.0);
    CHECK(A(1, 0) == 4.0);
    CHECK(A(1, 2) == 6.0);

    const arma::mat R = T.get_matrix(10, 0);
    REQUIRE(R.n_rows == 1);
    REQUIRE(R.n_cols == 2);
    CHECK(R(0, 0) == 6.0);
    CHECK(R(0, 1) == 0.0);

    const arma::mat C = T.get_matrix(11, 0);
    REQUIRE(C.n_rows == 3);
    REQUIRE(C.n_cols == 1);
    CHECK(C(2, 0) == 3.0);

    const arma::mat B = T.get_matrix(12, 0);
    REQUIRE(B.n_rows == 2);
    REQUIRE(B.n_cols == 2);
    CHECK(B(0, 0) == 6.0);
    CHECK(B(0, 1) == 0.0);
    CHECK(B(1, 0) == 10.0);
    CHECK(B(1, 1) == 0.0);

    CHECK(T.get_matrix(1, 0).is_empty());
    CHECK(T.get_matrix(20, 0).is_empty());

    CHECK(what([&] { T.get_matrix(13, 0); }) ==
          "mem.csv, line 16, column 'v' has rows of different length (row 2 has 1 entries, row 1 has 2).");
    CHECK(what([&] { T.get_matrix(19, 0); }) ==
          "mem.csv, line 22, column 'v' has rows of different length (row 2 has 3 entries, row 1 has 2).");
    CHECK(what([&] { T.get_matrix(14, 0); }) == "mem.csv, line 17, column 'v' has an empty entry (row 2, entry 1).");
    CHECK(what([&] { T.get_matrix(15, 0); }) == "mem.csv, line 18, column 'v' has an empty entry (row 2, entry 1).");
    CHECK(what([&] { T.get_matrix(16, 0); }) == "mem.csv, line 19, column 'v' has an empty entry (row 1, entry 2).");
    CHECK(what([&] { T.get_matrix(17, 0); }) ==
          "mem.csv, line 20, column 'v' has an entry that is not numeric (row 2, entry 1 ('x')).");
    const arma::mat Nm = T.get_matrix(18, 0);
    REQUIRE(Nm.n_rows == 2);
    CHECK(std::isnan(Nm(0, 0)));
    CHECK(Nm(1, 1) == INFINITY);
    CHECK(what([&] { T.get_matrix(18, 0, false, true); }).find("(row 1, entry 1 ('nan'))") != std::string::npos);
    CHECK(what([&] { T.get_matrix(18, 0, true, false); }).find("(row 2, entry 2 ('Inf'))") != std::string::npos);

    // Round trip of a matrix cell through the writer text
    arma::mat M(2, 3);
    M = {{0.1, -2.0, 1e-30}, {1.0 / 3.0, 4.0, 5e300}};
    qd_csv::table W;
    W.header = {"m"};
    W.add_row({qd_csv::format_matrix(M)});
    CHECK(qd_csv::format_matrix(M) == "0.1,-2,1e-30;0.3333333333333333,4,5e+300");
    const auto P = parse(written(W));
    const arma::mat M2 = P.get_matrix(0, 0);
    REQUIRE(M2.n_rows == 2);
    REQUIRE(M2.n_cols == 3);
    for (arma::uword i = 0; i < 6; ++i)
        CHECK(same_bits(M2(i), M(i)));
    CHECK(qd_csv::format_list({0.5, -1.0, 1e15}) == "0.5,-1,1e+15");
    CHECK(qd_csv::format_list({}) == "");
    CHECK(qd_csv::format_matrix(arma::mat()) == "");
}

TEST_CASE("qd_csv - a carriage return is part of CRLF only")
{
    // CR-only line ends, a CR inside a quoted cell, a CR inside an unquoted cell, a CR at the end of the text without LF:
    // an error with file and line (the text is not silently read as one line)
    const char *msg = ": a carriage return that is not part of a line end (line ends are LF or CRLF).";
    CHECK(what([] { parse("a,b\r1,2\r"); }) == std::string("mem.csv, line 1") + msg);
    CHECK(what([] { parse("a,b\n1,2\r3,4\n"); }) == std::string("mem.csv, line 2") + msg);
    CHECK(what([] { parse("a,b\r\n\"x\ry\",2\r\n"); }) == std::string("mem.csv, line 2") + msg);
    CHECK(what([] { parse("a,b\r\n1\r2,3\r\n"); }) == std::string("mem.csv, line 2") + msg);
    CHECK(what([] { parse("a,b\r\n1,2\r"); }) == std::string("mem.csv, line 2") + msg);
    CHECK(what([] { parse("a,b\n\r\n1,2\r\r\n"); }) == std::string("mem.csv, line 3") + msg); // CR CR LF
    CHECK(what([] { parse("# one\r# two\ra,b\n"); }) == std::string("mem.csv, line 1") + msg);  // also in comments
    CHECK(what([] { parse("a,b\n \r \n1,2\n"); }) == std::string("mem.csv, line 2") + msg);   // and blank lines

    // CRLF is fine everywhere, also with a quoted last cell and blank and comment lines
    const auto T = parse("a,b\r\n\"x\",\"y\"\r\n\r\n# c\r\n1,2\r\n");
    REQUIRE(T.n_rows() == 2);
    CHECK(T.cell(0, 1) == "y");
    CHECK(T.cell(1, 1) == "2");
    CHECK(T.line(1) == 5);
}

TEST_CASE("qd_csv - error messages carry the line of the file")
{
    // Line numbers count comments and blank lines, also after a BOM and with CRLF
    CHECK(what([] { parse("# c\r\n\r\na,b\r\n1,2\r\n# c\r\n\"x,2\r\n"); }) ==
          "mem.csv, line 6: a quoted cell is not closed (a quoted cell cannot contain a line break).");
    CHECK(what([] { parse("a,b\n\"x\" y,2\n"); }) == "mem.csv, line 2: text after the closing quote of a cell.");
    CHECK(what([] { parse("a,b\n1,\"x\"\"\n"); }) ==
          "mem.csv, line 2: a quoted cell is not closed (a quoted cell cannot contain a line break).");
    CHECK(what([] { parse("a,b\n1,\"x\n2,3\"\n"); }) ==
          "mem.csv, line 2: a quoted cell is not closed (a quoted cell cannot contain a line break).");
    CHECK(what([] { parse(""); }) == "mem.csv: there is no header line.");
    CHECK(what([] { parse("\xEF\xBB\xBF"); }) == "mem.csv: there is no header line.");
    CHECK(what([] { parse("# only\n\n   \n"); }) == "mem.csv: there is no header line.");

    // A cell error after rows that were skipped
    const auto T = parse("# x\na,b\n\n1,2\n# y\n3,oops\n");
    CHECK(what([&] { T.get_double(1, 1); }) == "mem.csv, line 6, column 'b' is not numeric ('oops').");
}

TEST_CASE("qd_csv - writer: quoting rules and exact text")
{
    qd_csv::table T;
    T.source = "w.csv";
    T.header = {"name", "text", "num"};
    T.add_row({"plain", "x", "1"});
    T.add_row({"a,b", "say \"hi\"", "2"});
    T.add_row({" lead", "trail ", "semi;colon"});
    T.add_row({"#c", "", ""});
    T.add_row({"x", "#y", "tab\t"});
    T.add_row({"", "\xC3\xA9", "end"});
    T.add_row({"only"}); // padded with empty cells
    CHECK(written(T) == "name,text,num\n"
                        "plain,x,1\n"
                        "\"a,b\",\"say \"\"hi\"\"\",2\n"
                        "\" lead\",\"trail \",\"semi;colon\"\n"
                        "\"#c\",,\n"
                        "x,\"#y\",\"tab\t\"\n"
                        ",\xC3\xA9,end\n"
                        "only,,\n");

    // The reader gives the cells back
    const auto R = parse(written(T));
    REQUIRE(R.n_rows() == T.n_rows());
    REQUIRE(R.header == T.header);
    for (size_t r = 0; r < T.n_rows(); ++r)
        for (size_t c = 0; c < T.n_cols(); ++c)
        {
            INFO("cell " << r << ", " << c);
            CHECK(R.cell(r, c) == T.cell(r, c));
        }

    // Header names are quoted by the same rules
    qd_csv::table H;
    H.header = {"a b", "c,d", " e", "f\"g", "#h"};
    CHECK(written(H) == "a b,\"c,d\",\" e\",\"f\"\"g\",\"#h\"\n");
    CHECK(parse(written(H)).header == H.header);

    // One column: an empty cell is quoted, else its line would be blank
    qd_csv::table S;
    S.header = {"v"};
    S.add_row({"1"});
    S.add_row({""});
    S.add_row({"x"});
    CHECK(written(S) == "v\n1\n\"\"\nx\n");
    const auto SR = parse(written(S));
    REQUIRE(SR.n_rows() == 3);
    CHECK(SR.cell(1, 0) == "");
    CHECK(SR.cell(2, 0) == "x");

    // Two columns with two empty cells are not blank
    qd_csv::table E;
    E.header = {"a", "b"};
    E.add_row({"", ""});
    CHECK(written(E) == "a,b\n,\n");
    CHECK(parse(written(E)).n_rows() == 1);

    // Row lines of a built table are the lines of the text it is written as
    CHECK(T.line(0) == 2);
    CHECK(T.line(6) == 8);

    // A line break cannot be written
    qd_csv::table X;
    X.header = {"a", "b"};
    X.add_row({"ok", "line\nbreak"});
    CHECK_THROWS_AS(written(X), std::invalid_argument);
    qd_csv::table Y;
    Y.header = {"a\r"};
    CHECK_THROWS_AS(written(Y), std::invalid_argument);
}

TEST_CASE("qd_csv - writer: number format")
{
    // Text and value pairs: the fewest of 15, 16, 17 significant digits that read back (Python as the oracle)
    struct case_t
    {
        double v;
        const char *text;
    };
    const case_t cases[] = {{0.0, "0"},
                            {-0.0, "-0"},
                            {1.0, "1"},
                            {0.1, "0.1"},
                            {0.5, "0.5"},
                            {100.0, "100"},
                            {1e5, "100000"},
                            {1e14, "100000000000000"},
                            {1e15, "1e+15"},
                            {123456789012345.0, "123456789012345"},
                            {1.0 / 3.0, "0.3333333333333333"},
                            {2.0 / 3.0, "0.6666666666666666"},
                            {0.1 + 0.2, "0.30000000000000004"},
                            {1e-5, "1e-05"},
                            {1e-4, "0.0001"},
                            {5e-324, "4.94065645841247e-324"},
                            {1.7976931348623157e308, "1.7976931348623157e+308"},
                            {2.2250738585072014e-308, "2.2250738585072014e-308"},
                            {3.141592653589793, "3.141592653589793"},
                            {-0.5235987755982988, "-0.5235987755982988"},
                            {1e23, "1e+23"},
                            {123456789012345678.0, "1.2345678901234568e+17"},
                            {4.35, "4.35"},
                            {-2.5e300, "-2.5e+300"},
                            {299792458.0, "299792458"},
                            {1.5e-7, "1.5e-07"},
                            {1e-320, "9.99988867182683e-321"}};
    for (const case_t &c : cases)
    {
        INFO("expected '" << c.text << "'");
        CHECK(qd_csv::format_number(c.v) == c.text);
        CHECK(same_bits(std::strtod(c.text, nullptr), c.v)); // the C library as the reader (C locale)
    }
    CHECK(qd_csv::format_number(NAN) == "NaN");
    CHECK(qd_csv::format_number(INFINITY) == "Inf");
    CHECK(qd_csv::format_number(-INFINITY) == "-Inf");

    // Fixed digits (angle columns): 15 significant digits, not more
    CHECK(qd_csv::format_number(0.1 + 0.2, 15, 15) == "0.3");
    CHECK(qd_csv::format_number(0.5235987755982988, 15, 15) == "0.523598775598299");
    CHECK(qd_csv::format_number(-3.141592653589793, 15, 15) == "-3.14159265358979");
    CHECK(qd_csv::format_number(1.0 / 3.0, 15, 15) == "0.333333333333333");
    CHECK(qd_csv::format_number(1.0 / 3.0, 17, 17) == "0.33333333333333331");
    CHECK(qd_csv::format_number(1.0 / 3.0, 3, 5) == "0.33333");     // the digits are limited by the range
    CHECK(qd_csv::format_number(0.5, 0, 0) == "0.5");               // limits are clamped to 1 .. 17
    CHECK(qd_csv::format_number(1.0 / 3.0, 40, 99) == "0.33333333333333331");

    // Read back (with strtod, independent of the unit): 15 digits give an error below half a unit of the 15th digit, which
    // is 5e-15 for the angles of one turn; the half-degree grid in degrees is bit-identical
    for (int k = -720; k <= 720; ++k)
    {
        const double deg = 0.5 * k;
        CHECK(same_bits(std::strtod(qd_csv::format_number(deg, 15, 15).c_str(), nullptr), deg));
    }
    for (int k = -100; k <= 100; ++k)
    {
        const double rad = 0.0314159265358979323846 * k + 1e-3 * (k % 7);
        CHECK(std::abs(std::strtod(qd_csv::format_number(rad, 15, 15).c_str(), nullptr) - rad) < 1e-14);
    }

    // Random bit patterns (fixed seed): every finite double reads back bit-identical
    std::mt19937_64 gen(12345);
    int n_checked = 0;
    for (int k = 0; k < 20000; ++k)
    {
        const uint64_t bits = gen();
        double v;
        std::memcpy(&v, &bits, sizeof(double));
        if (!std::isfinite(v))
            continue;
        REQUIRE(same_bits(std::strtod(qd_csv::format_number(v).c_str(), nullptr), v));
        ++n_checked;
    }
    CHECK(n_checked > 19000);
    for (int k = 0; k < 5000; ++k) // moderate values too: decimal-like and fractions
    {
        const double v = (double)(int)(gen() % 2000001 - 1000000) / (double)(1 + gen() % 997);
        REQUIRE(same_bits(std::strtod(qd_csv::format_number(v).c_str(), nullptr), v));
    }
}

TEST_CASE("qd_csv - numbers do not depend on the global locale")
{
    // Oracle: the same texts in the classic locale. The locale must have a decimal comma, so that a locale-dependent
    // conversion or format would read '-6.28' as -6 or write '0,1'.
    struct locale_guard
    {
        std::string c_name = std::setlocale(LC_ALL, nullptr);
        std::locale cpp_locale;
        ~locale_guard()
        {
            std::setlocale(LC_ALL, c_name.c_str());
            std::locale::global(cpp_locale);
        }
    };

    std::string used, decimal_point;
    double v1 = 0.0, v2 = 0.0, v3 = 0.0;
    std::string t1, t2, t3, file_text;
    qd_csv::table T;
    {
        locale_guard guard;
        for (const char *name : {"de_DE.UTF-8", "de_DE.utf8", "en_DK.UTF-8", "en_DK.utf8", "German_Germany.1252"})
        {
            try
            {
                std::locale loc(name);
                if (std::setlocale(LC_ALL, name) == nullptr)
                    continue;
                std::locale::global(loc);
                used = name;
                break;
            }
            catch (const std::runtime_error &)
            {
            }
        }
        if (!used.empty())
        {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%.1f", 1.5);
            decimal_point = buf;
            qd_csv::parse_number("-6.28", v1);
            qd_csv::parse_number("1.5e3", v2);
            qd_csv::parse_number("12345678.901234567", v3);
            t1 = qd_csv::format_number(0.1);
            t2 = qd_csv::format_number(-2.5e-7);
            t3 = qd_csv::format_number(1.0 / 3.0);
            T.header = {"v"};
            T.add_row({qd_csv::format_list({0.25, 1000.5})});
            file_text = written(T);
            const auto P = parse("a,b\n\"1.25,2.5\",3.75\n");
            CHECK(P.get_list(0, 0) == std::vector<double>{1.25, 2.5});
            CHECK(P.get_double(0, 1) == 3.75);
        }
    }
    if (used.empty())
        SKIP("no locale with a decimal comma is installed");

    CHECK(decimal_point == "1,5"); // the locale is a decimal-comma locale
    CHECK(v1 == -6.28);
    CHECK(v2 == 1500.0);
    CHECK(v3 == 12345678.901234567);
    CHECK(t1 == "0.1");
    CHECK(t2 == "-2.5e-07");
    CHECK(t3 == "0.3333333333333333");
    CHECK(file_text == "v\n\"0.25,1000.5\"\n");
}

TEST_CASE("qd_csv - files")
{
    // Read what was written; the text is written with LF
    qd_csv::table T;
    T.header = {"id", "label", "x"};
    T.add_row({"1", "first, with comma", "0.1"});
    T.add_row({"2", "second \"quoted\"", "-1e-30"});
    {
        temp_file f("test_qd_csv_rw.csv");
        qd_csv::write(f.fn, T);
        std::ifstream in(f.fn, std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        CHECK(text == "id,label,x\n1,\"first, with comma\",0.1\n2,\"second \"\"quoted\"\"\",-1e-30\n");

        const auto R = qd_csv::read(f.fn);
        CHECK(R.source == f.fn);
        REQUIRE(R.n_rows() == 2);
        CHECK(R.cell(0, 1) == "first, with comma");
        CHECK(R.cell(1, 1) == "second \"quoted\"");
        CHECK(R.get_double(1, 2) == -1e-30);

        // Writing again replaces the file
        qd_csv::table E;
        E.header = {"z"};
        qd_csv::write(f.fn, E);
        const auto R2 = qd_csv::read(f.fn);
        CHECK(R2.n_rows() == 0);
        CHECK(R2.header == std::vector<std::string>{"z"});
    }

    // Messages of a file carry its name
    {
        temp_file f("test_qd_csv_err.csv", "a,b\n1,2\n1,2,3\n");
        CHECK(what([&] { qd_csv::read(f.fn); }) == "test_qd_csv_err.csv, line 3: the row has 3 cells, the header has 2 columns.");
    }
    {
        temp_file f("test_qd_csv_empty.csv", "");
        CHECK(what([&] { qd_csv::read(f.fn); }) == "test_qd_csv_empty.csv: there is no header line.");
    }
    CHECK_THROWS_AS(qd_csv::read("test_qd_csv_does_not_exist.csv"), std::runtime_error);
    CHECK_THROWS_AS(qd_csv::read("."), std::runtime_error); // a folder
    CHECK_THROWS_AS(qd_csv::write("test_qd_csv_no_folder/x.csv", T), std::runtime_error);
}

TEST_CASE("qd_csv - input validation")
{
    const auto T = parse("a,b\n1,2\n");

    // Indices outside the table
    CHECK_THROWS_AS(T.cell(1, 0), std::out_of_range);
    CHECK_THROWS_AS(T.cell(0, 2), std::out_of_range);
    CHECK_THROWS_AS(T.cell((size_t)-1, 0), std::out_of_range);
    CHECK_THROWS_AS(T.line(1), std::out_of_range);
    CHECK_THROWS_AS(T.get_double(0, 2), std::out_of_range);
    CHECK_THROWS_AS(T.get_double(5, 0), std::out_of_range);
    CHECK_THROWS_AS(T.get_list(0, 9), std::out_of_range);
    CHECK_THROWS_AS(T.get_matrix(3, 0), std::out_of_range);
    CHECK(T.find("") == qd_csv::table::npos);
    CHECK(T.find("c") == qd_csv::table::npos);

    // An empty table
    const qd_csv::table N;
    CHECK(N.n_rows() == 0);
    CHECK(N.n_cols() == 0);
    CHECK(N.find("a") == qd_csv::table::npos);
    CHECK_THROWS_AS(N.cell(0, 0), std::out_of_range);

    // Rows that do not fit the header
    qd_csv::table W;
    CHECK_THROWS_AS(W.add_row({"1"}), std::invalid_argument); // no header yet
    W.header = {"a", "b"};
    CHECK_THROWS_AS(W.add_row({"1", "2", "3"}), std::invalid_argument);
    CHECK(W.n_rows() == 0);
    W.add_row(std::vector<std::string>{"x", "y"});
    const char *raw[] = {"p", "q"};
    W.add_row(raw);
    W.add_row(std::vector<std::string_view>{});
    REQUIRE(W.n_rows() == 3);
    CHECK(W.cell(0, 1) == "y");
    CHECK(W.cell(1, 0) == "p");
    CHECK(W.cell(2, 0) == "");

    // The header must not change its length once there are rows: no read past the stored cells
    qd_csv::table G;
    G.header = {"a"};
    G.add_row({"x"});
    G.header.push_back("b");
    CHECK_THROWS_AS(G.cell(0, 1), std::out_of_range);
    CHECK_THROWS_AS(G.get_double(0, 1), std::out_of_range);
    CHECK_THROWS_AS(G.add_row({"y", "z"}), std::invalid_argument);
    CHECK(G.cell(0, 0) == "x");
    G.header.resize(1);
    G.add_row({"y"});
    CHECK(G.cell(1, 0) == "y");
    G.header.clear(); // fewer columns than the cells of the rows
    CHECK_THROWS_AS(G.cell(0, 0), std::out_of_range);
    CHECK_THROWS_AS(G.add_row({}), std::invalid_argument);

    // The writer needs a column that can be read back: no columns, and one column without a name, throw
    qd_csv::table Z;
    CHECK_THROWS_AS(written(Z), std::invalid_argument);
    Z.header = {""};
    CHECK_THROWS_AS(written(Z), std::invalid_argument);
    Z.header = {"", ""}; // two empty names are a header that reads back (with the option for equal names)
    CHECK(written(Z) == ",\n");
    qd_csv::options dup;
    dup.allow_duplicate_header = true;
    CHECK(parse(written(Z), dup).header == Z.header);

    // The text of a table survives a move (the cells are offsets, not pointers)
    qd_csv::table Moved = std::move(W);
    CHECK(Moved.cell(0, 0) == "x");
    CHECK(Moved.cell(1, 1) == "q");
}

TEST_CASE("qd_csv - a table of 100,000 rows")
{
    const size_t n_row = 100000;
    qd_csv::table T;
    T.source = "big.csv";
    T.header = {"id", "name", "x", "y", "z", "angles", "note"};
    T.reserve(n_row, n_row * 64);
    for (size_t r = 0; r < n_row; ++r)
        T.add_row({std::to_string(r), "site_" + std::to_string(r % 977) + (r % 5 == 0 ? ",x" : ""),
                   qd_csv::format_number(0.001 * (double)r), qd_csv::format_number(-1.5 * (double)r),
                   qd_csv::format_number(1.0 / (double)(r + 1)), "0,90;180,270", r % 3 == 0 ? "" : "n"});

    temp_file f("test_qd_csv_big.csv");
    qd_csv::write(f.fn, T);
    const auto R = qd_csv::read(f.fn);

    REQUIRE(R.n_rows() == n_row);
    REQUIRE(R.n_cols() == 7);
    for (size_t r : {(size_t)0, (size_t)1, (size_t)5, (size_t)976, (size_t)54321, n_row - 1})
    {
        INFO("row " << r);
        CHECK(R.cell(r, 0) == std::to_string(r));
        CHECK(R.cell(r, 1) == "site_" + std::to_string(r % 977) + (r % 5 == 0 ? ",x" : ""));
        CHECK(R.get_double(r, 2) == 0.001 * (double)r);
        CHECK(R.get_double(r, 3) == -1.5 * (double)r);
        CHECK(R.get_double(r, 4) == 1.0 / (double)(r + 1));
        CHECK(R.cell(r, 6) == (r % 3 == 0 ? "" : "n"));
        CHECK(R.line(r) == r + 2);
    }
    const arma::mat A = R.get_matrix(77, 5);
    REQUIRE(A.n_rows == 2);
    CHECK(A(1, 1) == 270.0);
}

TEST_CASE("qd_csv - material table of obj_file_read")
{
    // The material reader of obj_file_read is built on the unit: quoting, comments, BOM and CRLF are accepted, a number
    // must be one finite decimal number, long rows and equal column names are tolerated, errors name file and line
    auto read_table = [](const std::string &text, std::vector<std::string> &names,
                         std::unordered_map<std::string, std::vector<double>> &prop)
    {
        temp_file f("test_qd_csv_mtl.csv", text);
        quadriga_lib::obj_file_read<double>("", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, f.fn,
                                            nullptr, &names, &prop);
    };
    std::vector<std::string> names;
    std::unordered_map<std::string, std::vector<double>> prop;

    // Quotes: a name with a comma and a quote
    read_table("name,a,note\n\"wall, thick\",2.5,\"\"\n\"say \"\"x\"\"\",\"4\",\n", names, prop);
    CHECK(names == std::vector<std::string>{"wall, thick", "say \"x\""});
    CHECK(prop["a"] == std::vector<double>{2.5, 4.0});
    CHECK(prop["note"] == std::vector<double>{0.0, 0.0}); // empty cells are 0

    // Comment lines are no rows, also before the header; BOM and CRLF
    read_table("\xEF\xBB\xBF# materials\r\nname,a,b\r\n# not a material\r\nair,1,0\r\n\r\n   # indented\r\nwall,5.24,0.5", names,
               prop);
    CHECK(names == std::vector<std::string>{"air", "wall"});
    CHECK(prop["a"] == std::vector<double>{1.0, 5.24});
    CHECK(prop["b"] == std::vector<double>{0.0, 0.5});

    // Long rows are ignored, equal column names are tolerated (the later column is the one in the output)
    read_table("name,a\nr,1,99,100\ns,2\n", names, prop);
    CHECK(names == std::vector<std::string>{"r", "s"});
    CHECK(prop["a"] == std::vector<double>{1.0, 2.0});
    CHECK(prop.size() == 1);
    read_table("name,a,a\nr,1,2\n", names, prop);
    CHECK(prop["a"] == std::vector<double>{2.0});

    // A trailing comma in the header (spreadsheet exports) adds no column; an empty name inside the header is an error
    read_table("name,a,\nr,1\ns,2,\n", names, prop);
    CHECK(names == std::vector<std::string>{"r", "s"});
    CHECK(prop["a"] == std::vector<double>{1.0, 2.0});
    CHECK(prop.size() == 1);
    read_table("name,a,,\nr,1,,\n", names, prop);
    CHECK(prop.size() == 1);
    read_table("name,a, , \"\"\nr,1\n", names, prop);
    CHECK(prop.size() == 1);
    CHECK(what([&] { read_table("name,,a\nr,1,2\n", names, prop); }) ==
          "Material file 'test_qd_csv_mtl.csv' has an empty column name in the header.");
    CHECK(what([&] { read_table(",a\nr,1\n", names, prop); }) ==
          "Material file 'test_qd_csv_mtl.csv' is missing required column 'name'.");

    // Hexadecimal numbers, which std::stod accepted, and other texts that are not one decimal number
    for (const char *cell : {"0x10", "0x1p3", "1e", "1,5", "nan", "inf", "-Inf", "1e999", "1 2"})
    {
        INFO("cell '" << cell << "'");
        const std::string msg = what([&] { read_table(std::string("name,a\nr,\"") + cell + "\"\n", names, prop); });
        CHECK(msg.find("test_qd_csv_mtl.csv, line 2, column 'a' is ") == 0);
    }

    // Errors name file and line
    CHECK(what([&] { read_table("name,a\nr,1\n\"s,2\n", names, prop); }) ==
          "test_qd_csv_mtl.csv, line 3: a quoted cell is not closed (a quoted cell cannot contain a line break).");
    CHECK(what([&] { read_table("# c\nname,a\nr,1\nr,2\n", names, prop); }) ==
          "Duplicate material name 'r' in 'test_qd_csv_mtl.csv'.");
    CHECK(what([&] { read_table("a,b\n1,2\n", names, prop); }) ==
          "Material file 'test_qd_csv_mtl.csv' is missing required column 'name'.");
    CHECK(what([&] { read_table("name,a\n# c\n,2\n", names, prop); }) ==
          "Material file 'test_qd_csv_mtl.csv' line 3 has an empty material name.");
    CHECK(what([&] { read_table("name,,a\nr,1,2\n", names, prop); }) ==
          "Material file 'test_qd_csv_mtl.csv' has an empty column name in the header.");
    CHECK(what([&] { read_table("name,a\n", names, prop); }) ==
          "Material file 'test_qd_csv_mtl.csv' contains no material rows.");
    CHECK(!what([&] { read_table("", names, prop); }).empty());
    CHECK(!what([&] { read_table("# only a comment\n", names, prop); }).empty());

    // The built-in table is read by the same code
    quadriga_lib::obj_file_read<double>("", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "", nullptr,
                                        &names, &prop);
    CHECK(names.size() == 25);
    CHECK(names[0] == "air");
    CHECK(names[9] == "itu_concrete");
    CHECK(prop["a"][9] == 5.24);
    CHECK(prop["att"][24] == 23.0);
}
