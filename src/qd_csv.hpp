// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Reader and writer for comma-separated tables with a header line (private, header-only)
//
// Purpose
// - Shared by the files that parse CSV (the material table in obj_file.cpp, the scenario parameter file reader for
//   its number syntax, the layout CSV files). A self-contained format layer with its own tests (test_qd_csv.cpp) that
//   several sections use, so it has a header of its own, as qd_hdf5.hpp, instead of a place in a qd_helpers_* header.
//
// Syntax
// - The first line that is neither blank nor a comment is the header. Lines whose first non-blank character is '#'
//   and blank lines are skipped. LF and CRLF line ends and a leading UTF-8 byte order mark are accepted.
// - Cells are separated by ','. Whitespace around an unquoted cell is removed. A cell is quoted with '"' (then it is
//   kept as it is, with '""' for a quote); it can contain commas and semicolons but no line break, and only blanks
//   may follow the closing quote. A quote inside an unquoted cell is an ordinary character.
// - Header names are case-sensitive and unique, unless the option 'allow_duplicate_header' is set (then 'find' returns
//   the first). A row with fewer cells than the header is padded with empty cells. A row with more cells is an error,
//   unless the option 'allow_long_rows' is set (then the extra cells are dropped).
// - Numbers: [+-] digits [. digits] [e|E [+-] digits] (at least one digit before the exponent), or [+-] Inf, or
//   [+-] NaN, in any case. The whole cell must be consumed, hexadecimal and other C notations are not numbers.
//   The conversion does not depend on the global locale.
// - List cell: numbers separated by ','. Matrix cell: rows separated by ';', numbers in a row separated by ','.
// - Errors are std::runtime_error with the source (file name), the line, and where it applies the column name.
//
// Storage: the table owns one copy of the text; a cell is an offset and a length into it, so that a file with several
// 100,000 rows does not allocate one string per cell.
//
// The writer quotes a cell only when needed (comma, semicolon, quote, leading or trailing blank, or '#' as first
// character, which would start a comment) and writes numbers with 15 significant digits, or 16 or 17 where 15 do not
// read back to the same double.

#ifndef qd_csv_hpp
#define qd_csv_hpp

#include <armadillo>
#include <clocale>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace qd_csv
{

// Result of 'parse_number'
enum class number_status : unsigned char
{
    ok,      // A number; 'v' is set
    invalid, // Not a number by the grammar
    range    // Beyond the range of a double
};

// Blank characters that are removed around unquoted cells and list entries ('\n' cannot occur inside a line)
inline bool is_blank(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f'; }

inline std::string_view trim(std::string_view s)
{
    while (!s.empty() && is_blank(s.front()))
        s.remove_prefix(1);
    while (!s.empty() && is_blank(s.back()))
        s.remove_suffix(1);
    return s;
}

// Parse one number token (grammar above); NaN is accepted only with 'allow_nan', Inf always
// - A number beyond the range of a double is 'range'; a number below the smallest subnormal is the 0 with its sign,
//   which is what strtod returns in the C locale
// - 'v' is written only for 'ok'
// - The conversion is strtod on the text checked here, with the decimal point of the current C locale; it is not
//   std::from_chars, which needs GLIBCXX_3.4.29 for double and is not in the libstdc++ shipped with MATLAB R2024a
inline number_status parse_number(std::string_view s, double &v, bool allow_nan = false)
{
    const size_t n = s.size();
    size_t i = 0;
    bool negative = false;
    if (i < n && (s[i] == '+' || s[i] == '-'))
        negative = s[i++] == '-';

    if (n - i == 3) // 'x | 0x20' maps a letter to its lower case and no other character to a letter
    {
        const char a = (char)(s[i] | 0x20), b = (char)(s[i + 1] | 0x20), c = (char)(s[i + 2] | 0x20);
        if (a == 'i' && b == 'n' && c == 'f')
        {
            v = negative ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
            return number_status::ok;
        }
        if (allow_nan && a == 'n' && b == 'a' && c == 'n')
        {
            v = std::numeric_limits<double>::quiet_NaN();
            return number_status::ok;
        }
    }

    // Decimal order of the first non-zero digit: digits before the point counted from it, minus the zeros after
    // the point before it; plus the exponent (saturated)
    long long order = 0;
    bool nonzero = false;
    size_t n_digit = 0;
    while (i < n && s[i] >= '0' && s[i] <= '9')
    {
        nonzero = nonzero || s[i] != '0';
        order += nonzero ? 1 : 0;
        ++i, ++n_digit;
    }
    if (i < n && s[i] == '.')
    {
        ++i;
        while (i < n && s[i] >= '0' && s[i] <= '9')
        {
            nonzero = nonzero || s[i] != '0';
            order -= nonzero ? 0 : 1;
            ++i, ++n_digit;
        }
    }
    if (n_digit == 0)
        return number_status::invalid;

    if (i < n && (s[i] == 'e' || s[i] == 'E'))
    {
        ++i;
        bool negative_exp = false;
        if (i < n && (s[i] == '+' || s[i] == '-'))
            negative_exp = s[i++] == '-';
        size_t n_exp = 0;
        long long exp = 0;
        while (i < n && s[i] >= '0' && s[i] <= '9')
        {
            exp = std::min(exp * 10 + (s[i] - '0'), 1000000000LL);
            ++i, ++n_exp;
        }
        if (n_exp == 0)
            return number_status::invalid;
        order += negative_exp ? -exp : exp;
    }
    if (i != n)
        return number_status::invalid;

    // The text is a plain decimal number. Zero, and orders far outside the range of a double, need no conversion.
    if (!nonzero || order < -330)
    {
        v = negative ? -0.0 : 0.0;
        return number_status::ok;
    }
    if (order > 310)
        return number_status::range;

    char local[64];
    std::string long_text;
    const char *text = local;
    // localeconv returns static data: calling it from several OpenMP threads is safe in practice as long as no thread
    // changes the locale at the same time, which the standard does not guarantee
    const char *point = std::localeconv()->decimal_point;
    if (point[0] == '.' && point[1] == '\0')
    {
        if (n < sizeof(local))
        {
            std::memcpy(local, s.data(), n);
            local[n] = '\0';
        }
        else
            long_text.assign(s.data(), n), text = long_text.c_str();
    }
    else // C locale with another decimal point, e.g. a comma
    {
        for (char c : s)
            if (c == '.')
                long_text += point;
            else
                long_text += c;
        text = long_text.c_str();
    }
    const double w = std::strtod(text, nullptr);
    if (std::isinf(w))
        return number_status::range;
    v = w;
    return number_status::ok;
}

// Number text that 'parse_number' reads back to the same double: the fewest of 'min_digits' to 'max_digits'
// significant digits (1 to 17); NaN is written as NaN, infinity as Inf and -Inf
// - 15 digits for values that are decimal in nature (integer and half degrees read back bit-identical); 16 or 17 where
//   15 do not read back. With min_digits = max_digits the digits are fixed.
inline std::string format_number(double v, int min_digits = 15, int max_digits = 17)
{
    if (std::isnan(v))
        return "NaN";
    if (std::isinf(v))
        return v < 0.0 ? "-Inf" : "Inf";
    min_digits = std::min(std::max(min_digits, 1), 17);
    max_digits = std::min(std::max(max_digits, min_digits), 17);

    // Same as in parse_number: safe from several threads as long as the locale is not changed meanwhile
    const char *point = std::localeconv()->decimal_point;
    const bool plain = point[0] == '.' && point[1] == '\0';
    char buf[64];
    std::string s;
    for (int digits = min_digits;; ++digits)
    {
        std::snprintf(buf, sizeof(buf), "%.*g", digits, v);
        s = buf;
        if (!plain) // snprintf uses the decimal point of the C locale
        {
            const size_t pos = s.find(point);
            if (pos != std::string::npos)
                s.replace(pos, std::strlen(point), ".");
        }
        if (digits >= max_digits)
            break;
        double w = 0.0;
        if (parse_number(s, w) == number_status::ok && std::memcmp(&v, &w, sizeof(double)) == 0)
            break;
    }
    return s;
}

// Numbers joined by ','
inline std::string format_list(const std::vector<double> &values, int min_digits = 15, int max_digits = 17)
{
    std::string s;
    for (size_t i = 0; i < values.size(); ++i)
        s += (i == 0 ? "" : ",") + format_number(values[i], min_digits, max_digits);
    return s;
}

// Matrix rows joined by ';', the numbers of a row by ','
inline std::string format_matrix(const arma::mat &M, int min_digits = 15, int max_digits = 17)
{
    std::string s;
    for (arma::uword r = 0; r < M.n_rows; ++r)
    {
        if (r != 0)
            s += ';';
        for (arma::uword c = 0; c < M.n_cols; ++c)
            s += (c == 0 ? "" : ",") + format_number(M(r, c), min_digits, max_digits);
    }
    return s;
}

// Options of the reader
struct options
{
    bool allow_long_rows = false;        // Drop the cells of a row beyond the header instead of throwing
    bool allow_duplicate_header = false; // Accept equal header names ('find' returns the first)
};

namespace detail
{
// Text of one number for the cell getters: nullptr if it is a number, else the reason
inline const char *check_number(std::string_view t, double &v, bool allow_nan, bool allow_inf)
{
    const number_status st = parse_number(t, v, allow_nan);
    if (st == number_status::range)
        return "is out of range";
    if (st != number_status::ok || (!allow_inf && std::isinf(v)))
        return "is not numeric";
    return nullptr;
}

inline std::string quoted(std::string_view t) { return "('" + std::string(t) + "')"; }
} // namespace detail

// A table with a header line; the cells are views into the buffer the table owns
class table
{
  public:
    std::string source;              // Name of the file, for the messages
    std::vector<std::string> header; // Column names

    size_t n_rows() const { return row_line.size(); }
    size_t n_cols() const { return header.size(); }

    // Cell text; throws std::out_of_range for an index outside the table. The view is valid until the table changes
    std::string_view cell(size_t row, size_t col) const
    {
        if (row >= row_line.size() || col >= header.size() || col >= stride) // 'stride' is the header size at the first row
            throw std::out_of_range("qd_csv::table: cell (" + std::to_string(row) + ", " + std::to_string(col) +
                                    ") is outside the table.");
        const ref &c = cells[row * stride + col];
        return std::string_view(buffer.data() + c.offset, c.length);
    }

    // Line number (1-based) of a row in the source text; for rows made by 'add_row' the line it has when written
    size_t line(size_t row) const
    {
        if (row >= row_line.size())
            throw std::out_of_range("qd_csv::table: row " + std::to_string(row) + " is outside the table.");
        return row_line[row];
    }

    // Index of the column with this name (the first one), or 'npos'
    size_t find(std::string_view name) const
    {
        for (size_t i = 0; i < header.size(); ++i)
            if (header[i] == name)
                return i;
        return npos;
    }
    static constexpr size_t npos = (size_t)-1;

    // Position of a cell for messages: "<source>, line <n>, column '<name>'"
    std::string where(size_t row, size_t col) const
    {
        return source + ", line " + std::to_string(line(row)) + ", column '" + header.at(col) + "'";
    }

    // Number in a cell; an empty cell gives 'if_empty'; throws if the cell is not one number
    double get_double(size_t row, size_t col, double if_empty = 0.0, bool allow_nan = true, bool allow_inf = true) const
    {
        const std::string_view t = cell(row, col);
        if (t.empty())
            return if_empty;
        double v = 0.0;
        if (const char *why = detail::check_number(t, v, allow_nan, allow_inf))
            throw std::runtime_error(where(row, col) + " " + why + " " + detail::quoted(t) + ".");
        return v;
    }

    // Numbers separated by ','; an empty cell gives an empty list; an empty entry is an error
    std::vector<double> get_list(size_t row, size_t col, bool allow_nan = true, bool allow_inf = true) const
    {
        const std::string_view t = cell(row, col);
        std::vector<double> out;
        if (!trim(t).empty())
            entries(t, out, row, col, "", allow_nan, allow_inf);
        return out;
    }

    // Matrix with rows separated by ';' and numbers in a row by ','; an empty cell gives an empty matrix; the rows
    // must have the same length
    arma::mat get_matrix(size_t row, size_t col, bool allow_nan = true, bool allow_inf = true) const
    {
        const std::string_view t = cell(row, col);
        if (trim(t).empty())
            return arma::mat();
        std::vector<std::vector<double>> rows;
        size_t pos = 0;
        for (size_t r = 1;; ++r)
        {
            const size_t end = std::min(t.find(';', pos), t.size());
            rows.emplace_back();
            entries(t.substr(pos, end - pos), rows.back(), row, col, "row " + std::to_string(r) + ", ", allow_nan,
                    allow_inf);
            if (rows.back().size() != rows[0].size())
                throw std::runtime_error(where(row, col) + " has rows of different length (row " + std::to_string(r) +
                                         " has " + std::to_string(rows.back().size()) + " entries, row 1 has " +
                                         std::to_string(rows[0].size()) + ").");
            if (end >= t.size())
                break;
            pos = end + 1;
        }
        arma::mat M((arma::uword)rows.size(), (arma::uword)rows[0].size());
        for (size_t r = 0; r < rows.size(); ++r)
            for (size_t c = 0; c < rows[r].size(); ++c)
                M((arma::uword)r, (arma::uword)c) = rows[r][c];
        return M;
    }

    // Append a row of text cells (any container of string, string_view or const char*), padded with empty cells;
    // more cells than columns throw std::invalid_argument. The header must be set before the first row and not
    // change its length afterwards (std::invalid_argument).
    template <class Container>
    void add_row(const Container &values)
    {
        if (header.empty())
            throw std::invalid_argument("qd_csv::table: the header must be set before rows are added.");
        if (!row_line.empty() && header.size() != stride)
            throw std::invalid_argument("qd_csv::table: the number of columns changed after rows were added.");
        const size_t n = (size_t)std::distance(std::begin(values), std::end(values));
        if (n > header.size())
            throw std::invalid_argument("qd_csv::table: a row has " + std::to_string(n) + " cells, the table has " +
                                        std::to_string(header.size()) + " columns.");
        stride = header.size();
        size_t added = 0;
        for (const auto &v : values)
        {
            const std::string_view t(v);
            cells.push_back({buffer.size(), t.size()});
            buffer.append(t.data(), t.size());
            ++added;
        }
        for (; added < header.size(); ++added)
            cells.push_back({0, 0});
        row_line.push_back(row_line.size() + 2);
    }
    void add_row(std::initializer_list<std::string_view> values) { add_row<std::initializer_list<std::string_view>>(values); }

    // Reserve space for the rows and the text to come
    void reserve(size_t n_rows_expected, size_t n_bytes_expected)
    {
        cells.reserve(n_rows_expected * header.size());
        row_line.reserve(n_rows_expected);
        buffer.reserve(n_bytes_expected);
    }

  private:
    // Numbers of 'part' separated by ',' appended to 'out'; 'where_in_cell' is "" or "row r, " for the messages
    void entries(std::string_view part, std::vector<double> &out, size_t row, size_t col, const std::string &where_in_cell,
                 bool allow_nan, bool allow_inf) const
    {
        size_t pos = 0;
        for (size_t k = 1;; ++k)
        {
            const size_t end = std::min(part.find(',', pos), part.size());
            const std::string_view e = trim(part.substr(pos, end - pos));
            double v = 0.0;
            if (e.empty())
                throw std::runtime_error(where(row, col) + " has an empty entry (" + where_in_cell + "entry " +
                                         std::to_string(k) + ").");
            if (const char *why = detail::check_number(e, v, allow_nan, allow_inf))
                throw std::runtime_error(where(row, col) + " has an entry that " + why + " (" + where_in_cell + "entry " +
                                         std::to_string(k) + " " + detail::quoted(e) + ").");
            out.push_back(v);
            if (end >= part.size())
                break;
            pos = end + 1;
        }
    }

    struct ref
    {
        size_t offset, length;
    };

    std::string buffer;           // The text; a parsed table has its cells unquoted in place
    std::vector<ref> cells;       // n_rows * stride
    std::vector<size_t> row_line; // Line of each row
    size_t stride = 0;            // Cells per row: the header size when the first row was added

    friend table parse(std::string text, const std::string &source, const options &opt);
};

// Parse the text of a table; 'source' names it in the messages
inline table parse(std::string text, const std::string &source, const options &opt = options())
{
    table T;
    T.source = source;
    T.buffer = std::move(text);

    char *b = &T.buffer[0];
    const size_t n = T.buffer.size();
    size_t pos = 0;
    if (n >= 3 && (unsigned char)b[0] == 0xEF && (unsigned char)b[1] == 0xBB && (unsigned char)b[2] == 0xBF)
        pos = 3;

    struct span
    {
        size_t offset, length;
    };
    std::vector<span> line_cells;
    bool have_header = false;

    for (size_t line_no = 1; pos < n; ++line_no)
    {
        const char *nl = (const char *)std::memchr(b + pos, '\n', n - pos);
        size_t end = nl == nullptr ? n : (size_t)(nl - b);
        const size_t next = nl == nullptr ? n : end + 1;
        const size_t first = pos;
        pos = next;
        if (nl != nullptr && end > first && b[end - 1] == '\r') // CRLF
            --end;

        // The text of an error starts with the file name and the line; built only when an error is thrown
        auto at = [&] { return source + ", line " + std::to_string(line_no) + ": "; };
        if (end > first && std::memchr(b + first, '\r', end - first) != nullptr)
            throw std::runtime_error(at() + "a carriage return that is not part of a line end (line ends are LF or CRLF).");

        size_t p = first;
        while (p < end && is_blank(b[p]))
            ++p;
        if (p == end || b[p] == '#') // blank line or comment
            continue;

        // Split the line into cells; a quoted cell is unquoted in place, which only moves text to the front
        line_cells.clear();
        for (;;)
        {
            while (p < end && is_blank(b[p]))
                ++p;
            if (p < end && b[p] == '"')
            {
                size_t w = p, r = p + 1;
                bool closed = false;
                while (r < end)
                {
                    if (b[r] != '"')
                        b[w++] = b[r++];
                    else if (r + 1 < end && b[r + 1] == '"')
                        b[w++] = '"', r += 2;
                    else
                    {
                        closed = true;
                        ++r;
                        break;
                    }
                }
                if (!closed)
                    throw std::runtime_error(at() + "a quoted cell is not closed (a quoted cell cannot contain a line break).");
                while (r < end && is_blank(b[r]))
                    ++r;
                if (r < end && b[r] != ',')
                    throw std::runtime_error(at() + "text after the closing quote of a cell.");
                line_cells.push_back({p, w - p});
                p = r;
            }
            else
            {
                const char *comma = (const char *)std::memchr(b + p, ',', end - p);
                const size_t sep = comma == nullptr ? end : (size_t)(comma - b);
                size_t cell_end = sep;
                while (cell_end > p && is_blank(b[cell_end - 1]))
                    --cell_end;
                line_cells.push_back({p, cell_end - p});
                p = sep; // at the comma or the end
            }
            if (p < end && b[p] == ',')
            {
                ++p;
                continue;
            }
            break;
        }

        if (!have_header)
        {
            have_header = true;
            for (const span &c : line_cells)
            {
                std::string name(b + c.offset, c.length);
                if (!opt.allow_duplicate_header)
                    for (const std::string &h : T.header)
                        if (h == name)
                            throw std::runtime_error(at() + "the column name '" + name + "' is used twice in the header.");
                T.header.push_back(std::move(name));
            }
            T.stride = T.header.size();
            continue;
        }

        if (line_cells.size() > T.header.size() && !opt.allow_long_rows)
            throw std::runtime_error(at() + "the row has " + std::to_string(line_cells.size()) +
                                     " cells, the header has " + std::to_string(T.header.size()) + " columns.");
        for (size_t c = 0; c < T.header.size(); ++c)
            if (c < line_cells.size())
                T.cells.push_back({line_cells[c].offset, line_cells[c].length});
            else
                T.cells.push_back({0, 0});
        T.row_line.push_back(line_no);
    }

    if (!have_header)
        throw std::runtime_error(source + ": there is no header line.");
    return T;
}

// Read a table from a file
inline table read(const std::string &fn, const options &opt = options())
{
    std::error_code ec;
    if (std::filesystem::is_directory(fn, ec)) // opening a folder succeeds on some systems
        throw std::runtime_error("Cannot read '" + fn + "': it is a folder.");
    std::ifstream f(fn, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file '" + fn + "'.");
    f.seekg(0, std::ios::end);
    const std::streamoff size = f.tellg();
    if (size < 0)
        throw std::runtime_error("Cannot read file '" + fn + "' (not a regular file).");
    f.seekg(0, std::ios::beg);
    std::string text((size_t)size, '\0');
    if (size > 0)
        f.read(&text[0], size);
    if (!f && size > 0)
        throw std::runtime_error("Cannot read file '" + fn + "'.");
    return parse(std::move(text), fn, opt);
}

namespace detail
{
// Cell text as written: quoted where the reader would otherwise change it; 'single_column' forces quotes on an empty
// cell, whose line would be blank
inline void write_cell(std::ostream &os, std::string_view t, bool single_column, const std::string &source)
{
    bool quote = single_column && t.empty();
    for (char c : t)
    {
        if (c == '\n' || c == '\r')
            throw std::invalid_argument("A cell contains a line break" + (source.empty() ? std::string() : " (" + source + ")") + ".");
        quote = quote || c == ',' || c == ';' || c == '"';
    }
    quote = quote || (!t.empty() && (is_blank(t.front()) || is_blank(t.back()) || t.front() == '#'));
    quote = quote || t.substr(0, 3) == "\xEF\xBB\xBF"; // the reader skips a byte order mark at the start of a file
    if (!quote)
    {
        os.write(t.data(), (std::streamsize)t.size());
        return;
    }
    os.put('"');
    for (char c : t)
    {
        if (c == '"')
            os.put('"');
        os.put(c);
    }
    os.put('"');
}
} // namespace detail

// Text of one cell as the writer writes it (quoted where the reader would otherwise change it); std::invalid_argument
// for a line break
inline std::string format_cell(std::string_view t)
{
    std::ostringstream os;
    detail::write_cell(os, t, false, std::string());
    return os.str();
}

// Write the header and the rows as text with LF line ends; throws std::invalid_argument for a cell with a line break,
// for a table without columns and for a table with one column whose name is empty (its header line would be blank)
inline void write(std::ostream &os, const table &T)
{
    if (T.n_cols() == 0 || (T.n_cols() == 1 && T.header[0].empty()))
        throw std::invalid_argument("A table needs a column, and one column needs a name (" + T.source + ").");
    const bool single = T.n_cols() == 1;
    for (size_t c = 0; c < T.n_cols(); ++c)
    {
        if (c != 0)
            os.put(',');
        detail::write_cell(os, T.header[c], false, T.source);
    }
    os.put('\n');
    for (size_t r = 0; r < T.n_rows(); ++r)
    {
        for (size_t c = 0; c < T.n_cols(); ++c)
        {
            if (c != 0)
                os.put(',');
            detail::write_cell(os, T.cell(r, c), single, T.source);
        }
        os.put('\n');
    }
}

// Write a table to a file (an existing file is replaced); throws std::runtime_error if the file cannot be written
inline void write(const std::string &fn, const table &T)
{
    std::ofstream f(fn, std::ios::binary | std::ios::trunc);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file '" + fn + "' for writing.");
    write(f, T);
    f.flush();
    if (!f)
        throw std::runtime_error("Cannot write file '" + fn + "'.");
}

} // namespace qd_csv

#endif
