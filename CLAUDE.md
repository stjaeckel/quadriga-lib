# Notes for coding sessions

Issues are tracked on GitHub (`stjaeckel/quadriga-lib`) with the `gh` CLI. When asked to make something an issue, to fix an issue, or to close one, follow the procedure in [CONTRIBUTING.md](CONTRIBUTING.md), section "For the maintainer and coding sessions". A problem a session finds on its own is drafted and shown first; it is filed only after the maintainer agrees.

Commit messages follow [CONTRIBUTING.md](CONTRIBUTING.md), section "Commit messages". Show the drafted message first and commit only when asked.

When asked to run a bug scan, follow [md/BUG_SCAN.md](md/BUG_SCAN.md) from phase 0. It runs only when the maintainer asks for it.

The rules for adding or changing an API function are in `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md` and `md/SPEC_PYTHON_API.md`.

Run the C++ test binary with `OMP_WAIT_POLICY=passive <build folder>/test_bin`, as `make test` does. With the default policy, waiting OpenMP threads keep spinning on their cores, and several test runs on one machine then slow each other down many times over. Timing measurements of OpenMP code are the exception: they run with the default policy, unless the measurement is about the passive policy. Start concurrent `test_bin` runs from separate checkouts: the tests write files with fixed names into the current folder and into `tests/data/`, and two runs that share either overwrite each other's files.
