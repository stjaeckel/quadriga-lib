// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Verification suite for quadriga_lib::conf_file_list.
//
// Pins down: which files of a directory are listed and in which order, the short names and their
// position, shadowing of a file name by an earlier directory, the ambiguity errors, directories that
// do not exist, and the three outputs.
// Left to other suites: the content of the files (test_conf_file_read).
//
// Oracles: the expected lists are written by hand next to the directory content of each case. The
// reference behaviour is QuaDRiGa's supported_scenarios.m with the changes Q9 and Q10 of the plan.
//
// Conventions: every case builds its own directories 'test_conf_file_list_*' in the working directory
// and removes them. Names are sorted byte-wise, so digits come before upper case before lower case.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_tools.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Diagnostic output. Comment out to silence the tables. The assertions run either way.
#define QD_TEST_VERBOSE

#ifdef QD_TEST_VERBOSE
#include <cstdio>
#define QD_DIAG(...) std::printf(__VA_ARGS__)
#else
#define QD_DIAG(...) ((void)0)
#endif

namespace
{
    typedef std::vector<std::string> strvec;

    // A directory that is created empty and removed with its content when it goes out of scope
    struct tmp_dir
    {
        std::string name;
        explicit tmp_dir(const std::string &n) : name(n)
        {
            std::filesystem::remove_all(name);
            std::filesystem::create_directory(name);
        }
        ~tmp_dir() { std::filesystem::remove_all(name); }
        void add(const std::string &file, const std::string &content = "") const
        {
            std::ofstream f(name + "/" + file, std::ios::binary);
            f << content;
        }
    };

    void print_list(const char *title, const strvec &scen, const strvec &files, const strvec &dirs)
    {
        QD_DIAG("\n# diag: conf_file_list %s\n", title);
        QD_DIAG("index,scenario,file_name,file_dir\n");
        for (size_t i = 0; i < scen.size(); ++i)
            QD_DIAG("%d,%s,%s,%s\n", (int)i, scen[i].c_str(), i < files.size() ? files[i].c_str() : "?",
                    i < dirs.size() ? dirs[i].c_str() : "?");
    }
}

TEST_CASE("conf_file_list - files ending in .conf are listed in byte-wise sorted order")
{
    tmp_dir d("test_conf_file_list_a");
    d.add("b.conf");
    d.add("A.conf");
    d.add("a.conf");
    d.add("10.conf");
    d.add("mmMAGIC_x.conf");
    d.add("QuaDRiGa_x.conf");
    d.add("3GPP_38.901_UMa.conf"); // dots in the name: only the last extension is removed
    d.add("notes.txt");
    d.add("old.conf.bak");
    d.add("UPPER.CONF"); // the extension is case-sensitive
    d.add("conf");
    std::filesystem::create_directory(d.name + "/folder.conf"); // not a file
    std::filesystem::create_directory(d.name + "/sub");
    { std::ofstream f(d.name + "/sub/nested.conf"); } // not recursive

    strvec scen = {"stale"}, files = {"stale"}, dirs = {"stale"};
    arma::uword n = quadriga_lib::conf_file_list({d.name}, scen, &files, &dirs);
    print_list("sorted order", scen, files, dirs);

    const strvec ref = {"10", "3GPP_38.901_UMa", "A", "QuaDRiGa_x", "a", "b", "mmMAGIC_x"};
    CHECK(n == 7);
    CHECK(scen == ref);
    REQUIRE(files.size() == 7);
    REQUIRE(dirs.size() == 7);
    for (size_t i = 0; i < 7; ++i)
    {
        CHECK(files[i] == ref[i] + ".conf");
        CHECK(dirs[i] == d.name);
    }
}

TEST_CASE("conf_file_list - short names follow their file, comments do not define one")
{
    tmp_dir d("test_conf_file_list_a");
    d.add("UMa_LOS.conf", "% ShortName = InComment\nDS_mu = -7 % ShortName = AfterValue\nShortName = Ul % x\n");
    d.add("UMa_NLOS.conf", "ShortName=Un\r\nDS_mu = -6\r\n\tShortName\t=\tUn2\r\n");
    d.add("Plain.conf", "DS_mu = -6\n");
    d.add("Broken.conf", "DS_mu = abc\nthis is not a conf line\nNumClusters = -1\n"); // content is not checked here

    strvec scen, files, dirs;
    arma::uword n = quadriga_lib::conf_file_list({d.name}, scen, &files, &dirs);
    print_list("short names", scen, files, dirs);

    CHECK(n == 7);
    CHECK(scen == strvec{"Broken", "Plain", "UMa_LOS", "Ul", "UMa_NLOS", "Un", "Un2"});
    CHECK(files == strvec{"Broken.conf", "Plain.conf", "UMa_LOS.conf", "UMa_LOS.conf", "UMa_NLOS.conf",
                          "UMa_NLOS.conf", "UMa_NLOS.conf"});
    CHECK(dirs == strvec(7, d.name));

    // Without parsing, only the files are listed
    n = quadriga_lib::conf_file_list({d.name}, scen, &files, &dirs, false);
    CHECK(n == 4);
    CHECK(scen == strvec{"Broken", "Plain", "UMa_LOS", "UMa_NLOS"});
    CHECK(files == strvec{"Broken.conf", "Plain.conf", "UMa_LOS.conf", "UMa_NLOS.conf"});
    CHECK(dirs == strvec(4, d.name));
}

TEST_CASE("conf_file_list - a file name defined by an earlier directory is skipped completely")
{
    tmp_dir a("test_conf_file_list_a"), b("test_conf_file_list_b");
    a.add("X.conf", "ShortName = Xs\n");
    a.add("Z.conf");
    b.add("X.conf", "ShortName = Other\n"); // shadowed: neither the file nor its short name is listed (Q10)
    b.add("Y.conf", "ShortName = Ys\n");
    b.add("A.conf");

    strvec scen, files, dirs;
    arma::uword n = quadriga_lib::conf_file_list({a.name, b.name}, scen, &files, &dirs);
    print_list("two directories, X.conf shadowed", scen, files, dirs);

    CHECK(n == 6);
    CHECK(scen == strvec{"X", "Xs", "Z", "A", "Y", "Ys"}); // directory order first, then name order
    CHECK(files == strvec{"X.conf", "X.conf", "Z.conf", "A.conf", "Y.conf", "Y.conf"});
    CHECK(dirs == strvec{a.name, a.name, a.name, b.name, b.name, b.name});

    // The other order of the directories lets the other X.conf win
    n = quadriga_lib::conf_file_list({b.name, a.name}, scen, &files, &dirs);
    CHECK(n == 6);
    CHECK(scen == strvec{"A", "X", "Other", "Y", "Ys", "Z"});
    CHECK(dirs == strvec{b.name, b.name, b.name, b.name, b.name, a.name});

    // A copy of a file with its short name in a second directory is not ambiguous (Q10)
    b.add("Z.conf", "ShortName = Zs\n");
    a.add("Z.conf", "ShortName = Zs\n");
    CHECK_NOTHROW(n = quadriga_lib::conf_file_list({a.name, b.name}, scen, &files, &dirs));
    CHECK(scen == strvec{"X", "Xs", "Z", "Zs", "A", "Y", "Ys"});

    // The same directory twice adds nothing
    strvec once, twice;
    quadriga_lib::conf_file_list({a.name}, once);
    quadriga_lib::conf_file_list({a.name, a.name}, twice);
    CHECK(once == twice);
}

TEST_CASE("conf_file_list - directories that do not exist are skipped")
{
    tmp_dir a("test_conf_file_list_a");
    a.add("X.conf");
    a.add("plain_file");

    strvec scen = {"stale"}, files = {"stale"}, dirs = {"stale"};

    SECTION("no directory")
    {
        CHECK(quadriga_lib::conf_file_list({}, scen, &files, &dirs) == 0);
        CHECK(scen.empty());
        CHECK(files.empty());
        CHECK(dirs.empty());
    }

    SECTION("missing directory and a file given as directory")
    {
        arma::uword n = 99;
        CHECK_NOTHROW(n = quadriga_lib::conf_file_list(
                          {"test_conf_file_list_missing", a.name + "/plain_file", a.name, a.name + "/X.conf"},
                          scen, &files, &dirs));
        CHECK(n == 1);
        CHECK(scen == strvec{"X"});
        CHECK(dirs == strvec{a.name});
    }

    SECTION("directory without conf files")
    {
        tmp_dir e("test_conf_file_list_b");
        CHECK(quadriga_lib::conf_file_list({e.name}, scen) == 0);
        CHECK(scen.empty());
    }

    SECTION("the directory is returned as given")
    {
        const std::string with_slash = a.name + "/";
        quadriga_lib::conf_file_list({with_slash}, scen, &files, &dirs);
        CHECK(scen == strvec{"X"});
        CHECK(dirs == strvec{with_slash});
    }
}

TEST_CASE("conf_file_list - an empty string is the current directory")
{
    // The working directory may hold other conf files, so only the presence of this one is asserted
    const std::string fn = "test_conf_file_list_cwd_Qz7.conf";
    {
        std::ofstream f(fn, std::ios::binary);
        f << "ShortName = test_conf_file_list_cwd_short\n";
    }

    strvec scen, files, dirs;
    arma::uword n = 0;
    try
    {
        n = quadriga_lib::conf_file_list({""}, scen, &files, &dirs);
    }
    catch (...)
    {
        std::filesystem::remove(fn);
        throw;
    }
    std::filesystem::remove(fn);

    CHECK(n == scen.size());
    REQUIRE(files.size() == scen.size());
    REQUIRE(dirs.size() == scen.size());
    int found = -1;
    for (size_t i = 0; i < scen.size(); ++i)
        if (scen[i] == "test_conf_file_list_cwd_Qz7")
            found = (int)i;
    REQUIRE(found >= 0);
    REQUIRE((size_t)found + 1 < scen.size());
    CHECK(files[found] == fn);
    CHECK(dirs[found] == "");
    CHECK(scen[found + 1] == "test_conf_file_list_cwd_short");
    CHECK(dirs[found + 1] == "");
}

TEST_CASE("conf_file_list - input validation")
{
    tmp_dir a("test_conf_file_list_a"), b("test_conf_file_list_b");
    a.add("M.conf", "ShortName = Ms\n");
    a.add("N.conf", "ShortName = Ns\n");

    strvec scen = {"marker"}, files = {"marker"}, dirs = {"marker"};
    const strvec marker = {"marker"};

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::conf_file_list({a.name, b.name}, scen, &files, &dirs));
        CHECK(scen == strvec{"M", "Ms", "N", "Ns"});
        CHECK_NOTHROW(quadriga_lib::conf_file_list({a.name, b.name}, scen)); // optional outputs omitted
        CHECK(scen == strvec{"M", "Ms", "N", "Ns"});
    }

    SECTION("short name equals an earlier file name")
    {
        a.add("N.conf", "ShortName = M\n");
        CHECK_THROWS_AS(quadriga_lib::conf_file_list({a.name, b.name}, scen, &files, &dirs), std::invalid_argument);
        CHECK(scen == marker); // outputs untouched
        CHECK(files == marker);
        CHECK(dirs == marker);
        CHECK_NOTHROW(quadriga_lib::conf_file_list({a.name, b.name}, scen, &files, &dirs, false)); // not parsed
        CHECK(scen == strvec{"M", "N"});
    }

    SECTION("short name equals its own file name")
    {
        a.add("N.conf", "ShortName = N\n");
        CHECK_THROWS_AS(quadriga_lib::conf_file_list({a.name}, scen), std::invalid_argument);
    }

    SECTION("short name equals an earlier short name, same file")
    {
        a.add("N.conf", "ShortName = Ns\nShortName = Ns\n");
        CHECK_THROWS_AS(quadriga_lib::conf_file_list({a.name}, scen), std::invalid_argument);
    }

    SECTION("short name equals an earlier short name, other directory")
    {
        b.add("P.conf", "ShortName = Ms\n");
        CHECK_THROWS_AS(quadriga_lib::conf_file_list({a.name, b.name}, scen, &files, &dirs), std::invalid_argument);
        CHECK(scen == marker);
    }

    SECTION("file name equals an earlier short name")
    {
        a.add("Ms.conf"); // sorted after M.conf, which defines the short name 'Ms'
        CHECK_THROWS_AS(quadriga_lib::conf_file_list({a.name}, scen), std::invalid_argument);
        b.add("Ns.conf");
        std::filesystem::remove(a.name + "/Ms.conf");
        CHECK_THROWS_AS(quadriga_lib::conf_file_list({a.name, b.name}, scen), std::invalid_argument);
        CHECK(scen == marker);
    }

    SECTION("malformed ShortName line")
    {
        a.add("N.conf", "ShortName =\n");
        CHECK_THROWS_AS(quadriga_lib::conf_file_list({a.name}, scen), std::invalid_argument);
        a.add("N.conf", "ShortName = two words\n");
        CHECK_THROWS_AS(quadriga_lib::conf_file_list({a.name}, scen), std::invalid_argument);
        CHECK(scen == marker);
    }
}
