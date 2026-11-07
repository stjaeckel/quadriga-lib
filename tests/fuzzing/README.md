# Fuzzing

This folder holds the fuzz tests of Quadriga-Lib: small programs, the targets, that feed generated input to the file readers and to API functions, and a runner that fuzzes them for a given time and writes a short report of what failed. Fuzzing complements the unit tests. The unit tests check chosen inputs and their results; fuzzing runs millions of mostly malformed inputs and checks only that the library answers each one with a result or a clean exception: no crash, no memory error, no undefined behaviour, no hang and no runaway memory use. It finds the inputs nobody thought of writing a test for, which matters most where the input comes from outside, such as files.

- [Fuzzing](#fuzzing)
  - [Running](#running)
  - [The output folder](#the-output-folder)
  - [Reading the report](#reading-the-report)
  - [Reproducing a finding](#reproducing-a-finding)
  - [Known findings](#known-findings)
  - [What is fuzzed](#what-is-fuzzed)
  - [Adding a target](#adding-a-target)
  - [How it works](#how-it-works)
  - [Folder layout](#folder-layout)

## Running

The fuzz build runs on Linux and needs Clang with its runtime libraries, which provide libFuzzer and the sanitizers (on Debian and Ubuntu the packages `clang` and `libclang-rt-<version>-dev`), CMake and Python 3; the runner uses the Python standard library only. Clang's OpenMP library (`libomp-<version>-dev`) is optional: without it the library builds single-threaded, which the fuzzing does not need anyway.

```bash
make test-fuzz
```

builds the library and the targets in `build_linux_fuzz`, fuzzes all targets for ten minutes and writes the report. A run overnight, with a chosen output folder and number of processes:

```bash
make test-fuzz FUZZ_TIME=8h FUZZ_JOBS=16 FUZZ_DIR=$HOME/quadriga-fuzz
```

| Variable         | Default               | Meaning                                                                         |
| :--------------- | :-------------------- | :------------------------------------------------------------------------------ |
| `FUZZ_TIME`      | `10m`                 | Fuzzing time of the run, e.g. `600`, `30m`, `8h`, `1h30m`                       |
| `FUZZ_JOBS`      | number of CPUs        | Fuzzing processes at the same time                                              |
| `FUZZ_DIR`       | `build_linux_fuzz_out`| Output folder: the corpus and the reports                                       |
| `FUZZ_BUILD_DIR` | `build_linux_fuzz`    | Build folder of the fuzz build                                                  |
| `FUZZ_CC`, `FUZZ_CXX` | `clang`, `clang++` | Compiler, e.g. `clang-21`, `clang++-21`                                       |
| `FUZZ_ARGS`      | empty                 | Further options of `run.py`, e.g. `FUZZ_ARGS='--targets fz_qdant fz_obj'`       |

The options of `run.py` (`python3 tests/fuzzing/run.py --help`) beyond those set by the variables are `--targets` (run only these targets), `--input-timeout` (seconds one input may take before it counts as a hang, default 10), `--rss-limit-mb` (memory one input may use before it counts as out-of-memory, default 2048), `--minimize-time` (seconds spent shrinking each new finding, default 60), `--max-replays` (failing inputs replayed per target, default 500; a defect that is easy to hit can leave thousands) and `--known` (the list of known findings, default `known.txt` in this folder).

Each fuzzing process runs one thread (`OMP_NUM_THREADS=1`); the parallel work comes from the processes. When there are at least as many processes as targets, all targets run for the whole time and the processes are shared out among them. Otherwise the targets run in batches and the time is split between the batches, with at least one minute per batch, so a short run of many targets on few processes can take longer than `FUZZ_TIME`. After the fuzzing time the runner replays and shrinks the failing inputs, which takes up to a few minutes per new finding. Ctrl-C or SIGTERM (for example from `timeout`) during the fuzzing ends it early, and the triage and the report follow as usual. Before the triage the runner writes a preliminary report, marked as such, so that a run stopped during the triage still leaves one.

The runner sets the sanitizer and OpenMP settings of the targets itself (`ASAN_OPTIONS`, `UBSAN_OPTIONS`, `OMP_NUM_THREADS`), because the triage depends on them; values set in the calling shell do not apply. The build, output and temp folders must not contain spaces, since libFuzzer passes them on to its workers unquoted; the runner refuses such paths.

## The output folder

```text
<FUZZ_DIR>/
├── corpus/<target>/          inputs that reached new code, kept from run to run
├── latest                    link to the newest run
└── runs/<date>_<time>/
    ├── report.md             the report
    ├── findings/F1/          per new finding: input, output.txt (sanitizer output), input.orig if minimized
    ├── artifacts/<target>/   every input that failed during fuzzing
    └── logs/<target>.log     libFuzzer's log
```

The corpus is what makes long fuzzing pay off: each run starts from the inputs that earlier runs found interesting and goes deeper from there, so several short runs add up much like one long run. Each target also writes its seed inputs into its corpus at the start of every run. Deleting `corpus/<target>` starts that target from its seeds again, which is useful after its input format changed a lot. `make clean` removes the default build folder and keeps the output folder; `make tidy` also removes the default output folder, and with it the corpus of all earlier runs.

## Reading the report

`report.md` is meant to be read in full: it lists what failed and leaves out everything that worked. It has these parts:

- **Header:** the commit (with a note if tracked files had uncommitted changes), start and end, the number of processes, the load before and after, the limits per input and the counts of new and known findings.
- **Targets:** per target the fuzzing time, the processes, the inputs run, the runs per second, the coverage (code edges reached; it compares runs of one target, not targets with each other), the corpus size at the start and at the end, and the number of failing inputs that were new, known or did not fail again.
- **Problems of the run:** a target whose seeds could not be written, whose fuzzer did not end normally or ran no input, or one where most fuzzing jobs ended at a failing input. Each failure ends a job, so such a target hardly explores past the failure; fixing the defect, or listing it in `known.txt` when it is undefined behaviour, lets it go further.
- **New findings:** one section per cause. Many failing inputs usually share one cause; they are grouped by their signature, which is the kind of failure, the first library function in the stack trace and its file (for example `heap-buffer-overflow in quadriga_lib::qdant_read (src/qd_arrayant_qdant.cpp)`). Each section gives the message of the sanitizer or of libFuzzer, how many inputs hit it, the smallest input after minimizing, the command that reproduces it, the stack frames in the source tree, the input as text if it is printable, and a command that recreates the input file, ready to paste into an issue.
- **Known findings:** causes listed in `known.txt`, counted only.
- **Inputs not replayed:** per target the failing inputs beyond `--max-replays`, which stay in `artifacts/` unexamined. The counts of inputs per finding are those of the replayed inputs.
- **Inputs that did not fail again:** inputs that failed while fuzzing but passed when run alone, most often a timeout under heavy load.

The kinds of failure:

| Kind | Meaning |
| :--- | :------ |
| `heap-buffer-overflow`, `heap-use-after-free`, `SEGV`, `stack-overflow`, ... | Memory error found by AddressSanitizer |
| `undefined-behavior` | Undefined behaviour found by UndefinedBehaviorSanitizer, e.g. a signed integer overflow or an invalid shift; the message names it |
| `uncaught exception <type>` | An exception reached the end of the input: one that does not derive from `std::exception`, or any exception thrown outside `fz::clean` (a mistake in the target) |
| `deadly signal` | The program ended on a signal without a sanitizer report, e.g. an `abort` |
| `timeout` | One input took longer than `--input-timeout` |
| `out-of-memory`, `out-of-memory (single allocation)` | One input used more than `--rss-limit-mb`, or asked for more in one allocation |
| `memory-leak` | Memory not freed at the end of the input, found by LeakSanitizer |
| `fuzz target exited`, `exit code <n>` | The program ended by itself during an input, or failed without any of the messages above |

A clean `std::exception` is the correct answer to invalid input and never shows up in the report. A timeout or an out-of-memory finding needs judgement: the library may need long or much memory for a large valid input, but a small file that makes a reader allocate gigabytes is a defect, because the reader can check the sizes it reads against the size of the file.

## Reproducing a finding

A target runs a single input when it is given a file, and prints the same sanitizer report. The "Reproduce" line of the report names the right target and sets the limits, the single thread and the sanitizer options of the run:

```bash
OMP_NUM_THREADS=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 build_linux_fuzz/fz_<target> -timeout=10 -rss_limit_mb=2048 -malloc_limit_mb=2048 <FUZZ_DIR>/latest/findings/F1/input
```

Without `halt_on_error=1`, undefined behaviour is printed and the input runs on, so libFuzzer does not count it as a failure. Where `llvm-symbolizer` is installed only with a version suffix, as on Debian and Ubuntu, the line also sets `ASAN_SYMBOLIZER_PATH`; without it the stack traces have no function names. The binary also runs under a debugger. The input of a file target is the file the reader gets, so it reproduces through the public reader function with any build of the library, for example in a small test run with `make test-asan`. The input of an API target is decoded into arguments by `fz::input`; to call the function outside the fuzz build, the arguments are read off the target's source, with a debugger or a print in a copy of the target, and written into the test by hand.

When a finding is fixed, its regression test goes into the normal test suite: a Catch2 test that writes the bytes of the input, or the few lines of it that matter, into a file (or passes the decoded arguments) and checks that the library now throws a clean error or gives the right result. The fuzz inputs themselves are not kept in the repository.

## Known findings

`known.txt` lists causes that are known and not fixed yet, one per line as `<signature> | <note>`, with the signature copied from the report. Inputs with a listed signature are counted under "Known findings" instead of coming back as new in every run. A line is added when a finding is filed and stays open, and removed with the fix, so that a regression shows up as new again. The signature names the function and the file but not the line, so it stays valid while the code around it changes.

Known undefined behaviour is also skipped while fuzzing: the runner turns its entries into an UndefinedBehaviorSanitizer suppressions file (`ubsan_suppressions.txt` in the run folder) for the named functions, and the report header lists them. Without that, a target whose inputs mostly hit one known case would end almost every fuzzing job there and never get further. Such inputs then no longer fail, so they do not appear under "Known findings". Memory errors, hangs and out-of-memory cases cannot be skipped this way; a target that keeps hitting one of them explores little until the defect is fixed.

## What is fuzzed

| Target | Input | Calls |
| :----- | :---- | :---- |
| `fz_qdant` | QDANT file | `qdant_read` (with layout, a second ID), `qdant_read_multi` |
| `fz_obj` | OBJ file | `obj_file_read` with all outputs |
| `fz_mtl` | MTL file | `obj_file_read` of a fixed OBJ file that uses the fuzzed MTL file |
| `fz_material_csv` | Material table (CSV) | `obj_file_read` of a fixed OBJ file with the fuzzed table, lenient and strict |
| `fz_conf` | Scenario parameter file | `conf_file_read` with all outputs, with and without the value checks |
| `fz_qrt` | QRT file | `qrt_file_parse`, `qrt_read_cache_init`, `qrt_file_read` for several origins and CIRs in both directions, `qrt_file_read_raw` |
| `fz_hdf5` | HDF5 channel file | `hdf5_read_layout`, `hdf5_read_channel`, `hdf5_read_dset_names`, `hdf5_read_dset` for the first slots |
| `fz_arrayant_generate` | Parameters | The `generate_arrayant_*` functions and `generate_speaker` |
| `fz_arrayant_interpolate` | Array antenna, angles | `arrayant::interpolate`, `arrayant_interpolate_multi` |
| `fz_arrayant_edit` | Array antenna | The `arrayant` member functions in a fuzzed order, the `arrayant_*_multi` functions, `qdant_write` and `qdant_read` |
| `fz_get_channels` | Array antennas, paths | `get_channels_spherical`, `get_channels_planar`, `get_channels_irs`, `get_channels_multifreq` |
| `fz_channel_tools` | Coefficients, delays | `quantize_delays`, the `baseband_freq_response` functions |
| `fz_hdf5_write` | Channel object | `channel` member functions, the HDF5 writers, then the HDF5 readers on the written file |
| `fz_sos` | Frequencies, phases, ACF tables | The `sos_*` functions |
| `fz_ray_intersect` | Rays, meshes, points | `ray_triangle_intersect`, `ray_point_intersect`, `subdivide_rays` |
| `fz_mesh_tools` | Triangle mesh | `point_inside_mesh`, `obj_overlap_test`, `subdivide_triangles`, the `triangle_mesh_*` functions, the mesh generators |
| `fz_point_cloud` | Points | The `point_cloud_*` functions |
| `fz_spreads` | Delays, angles, powers | `acdf`, `calc_delay_spread`, `calc_angular_spreads_sphere`, `calc_rician_k_factor`, `calc_cross_polarization_ratio` |
| `fz_path_tools` | Path coordinates | `coord2path`, `combine_irs_coord`, `path_to_tube`, `generate_diffraction_paths` |

The file targets write the fuzz input to a file and call the reader. The API targets turn the fuzz input into sizes, indices and values, with sizes that mostly fit together and sometimes do not, and call the functions with them; the comment at the top of each target lists in detail what it calls. Each API target picks `float` or `double` per input. Where the API allows an output argument to be the same object as an input, the targets also call it that way. Not fuzzed: the CUDA code, the MATLAB and Python interfaces, the correctness of results (the job of the unit tests) and thread safety.

## Adding a target

A target is one file `targets/fz_<name>.cpp`. CMake builds every such file into its own program, so the next `make test-fuzz` configures, builds and runs a new target without any registration; only the table above gets a row. The file includes `fuzz_target.hpp` and defines two functions:

```cpp
#include "fuzz_target.hpp"

// Seed inputs: valid examples of the input, written into the corpus at the start of each run
void fuzz_seeds(const fz::seeds &s)
{
    s.text("triangle.obj", "o Tri\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"); // given as text
    s.copy("tests/data/test_scene.obj");                                      // a file of the repository
}

// One input: call the library; a std::exception is the clean answer to invalid input
void fuzz_one(const uint8_t *data, size_t size)
{
    std::string fn = fz::write_file("in.obj", data, size);
    fz::clean([&] { arma::mat mesh; quadriga_lib::obj_file_read<double>(fn, &mesh); });
}
```

Seeds can also be written by the library's own writers through `s.path(<name>)`, for example `arrayant::qdant_write`. A target for API functions takes its arguments from the input with `fz::input`, which extends libFuzzer's `FuzzedDataProvider` by `size(max)`, `index(n)` (a valid index, one or two past the end, or the largest value), `value<T>()` (mostly moderate numbers, sometimes zero, NaN, infinity or extreme values), `col`, `mat` and `cube`; `fz::arrayant<T>(in)` builds an array antenna from it. The API targets run each input in `float` or `double`:

```cpp
#include "fuzz_target.hpp"

void fuzz_seeds(const fz::seeds &) {}

template <typename T>
static void run(fz::input &in)
{
    arma::Mat<T> points = in.mat<T>(in.size(32), in.ConsumeBool() ? 3 : in.size(4));
    fz::clean([&] { quadriga_lib::point_cloud_aabb<T>(&points); });
}

void fuzz_one(const uint8_t *data, size_t size)
{
    fz::input in(data, size);
    in.ConsumeBool() ? run<double>(in) : run<float>(in);
}
```

The targets follow a few rules, each for a reason:

- Every library call is wrapped in its own `fz::clean`, so that an exception in one call does not hide the calls after it.
- Seed inputs are given as text, written by the library or copied from files that are in the repository anyway (`s.copy` skips a missing file); no inputs are stored for fuzzing. A few small seeds that cover the main variants of a format are enough, since the fuzzer finds the rest. They are kept small (a few kB), because libFuzzer sets the largest input it tries from the largest seed.
- A reader that needs a file extension gets it through the name given to `fz::write_file`. Companion files, such as the MTL file of an OBJ file, are written into the same work folder (`fz::work_dir()`).
- The sizes in API targets are bounded, to about 64 elements, so that one input takes milliseconds and needs little memory; the bounds include 0 and 1. An allocation the caller asked for on purpose is not a defect, so a target does not let fuzzed sizes ask for gigabytes.
- Targets do not check results. A wrong result for a valid input belongs in a unit test, where the expected value is known.

A short run checks a new target: `make test-fuzz FUZZ_TIME=2m FUZZ_ARGS='--targets fz_<name>'`. A finding whose signature names a file in `targets/` (no library function was on the stack) is usually a mistake in the target. A target file kept outside the repository, for example to try it before adding it, can be built by hand from the repository root against the library of an existing fuzz build, with the compiler of that build; when Armadillo is not installed on the system, the fuzz build has unpacked it, and `-I build_linux_fuzz/armadillo-<version>/include` is added:

```bash
clang++ -std=c++17 -O1 -g -fno-omit-frame-pointer -fsanitize=fuzzer,address,undefined -fopenmp \
    -I tests/fuzzing -I include -D QD_SOURCE_DIR='"'$PWD'"' <folder>/fz_new.cpp build_linux_fuzz/libquadriga.a -o <folder>/fz_new
python3 tests/fuzzing/run.py --build-dir <folder> --out <output folder> --time 2m
```

Without Clang's OpenMP library, `-fopenmp` is left out.

## How it works

**Build.** `ENABLE_FUZZING=ON` compiles the whole library with libFuzzer's coverage instrumentation, AddressSanitizer and UndefinedBehaviorSanitizer at `-O1 -g`; MATLAB, Octave, Python and CUDA are off. UndefinedBehaviorSanitizer stays recoverable in the build, and the runner makes it stop the input at run time (`halt_on_error=1`), which is what allows known cases to be skipped with a suppressions file. Each target links libFuzzer, which provides `main`. AVX2 follows the `avx2` switch of the GNUmakefile; functions with an AVX2 kernel choose it at run time, so the CPU of the fuzzing machine decides which kernel is fuzzed.

**Fuzzing.** libFuzzer runs the target in its own process, many thousand inputs per second for most targets, and keeps the inputs that reach new code. The runner starts libFuzzer in fork mode with one or more worker processes per target: a worker that fails saves its input and is replaced, so one defect does not stop a long run. The input files of the targets go to a folder per process; the runner puts these folders on the RAM disk (`/dev/shm`) where available and removes them at the end, while a target run by hand leaves its folder `qd_fuzz_<pid>` in the system temp folder.

**Triage.** After the fuzzing time, the saved inputs are run alone, up to `--max-replays` per target, taken from each kind of failure in turn and at random within a kind. Random rather than smallest first, because the smallest inputs of an API target all decode to the same first cases and would hide the other causes. An input that passes is listed under "did not fail again". The output of the others gives the kind of failure and the stack trace, from which the signature is formed. For each signature that is not in `known.txt`, the smallest input is shrunk with libFuzzer's `-minimize_crash` for up to `--minimize-time` seconds and kept if it still fails with the same signature.

libFuzzer was chosen over AFL++ because it runs the inputs inside the target's process instead of starting the program for each one, which is many times faster for these readers, and because it ships with Clang. AFL++ can build the same targets, since both use the `LLVMFuzzerTestOneInput` interface.

## Folder layout

| File | Content |
| :--- | :------ |
| `README.md` | This description |
| `fuzz_target.hpp` | Common part of the targets: libFuzzer entry points, seed writing, input files, `fz::input`, `fz::arrayant`, `fz::clean` |
| `targets/fz_*.cpp` | The targets |
| `run.py` | Runner: schedules the targets, replays and groups the failing inputs, writes the report |
| `known.txt` | Known findings that are not fixed yet |
