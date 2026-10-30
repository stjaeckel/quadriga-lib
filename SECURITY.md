# Security policy

Please do not report security problems as a public issue. Use GitHub's private reporting instead: on the repository's **Security** tab, choose **Report a vulnerability** ([direct link](https://github.com/stjaeckel/quadriga-lib/security/advisories/new)). Only the maintainer sees the report.

Examples of security problems: a crafted input file (HDF5, OBJ, configuration file) that makes the library write outside its buffers or run code, or a build step that downloads and runs code.

Fixes go into the current version on `main`. Older releases are not patched.
