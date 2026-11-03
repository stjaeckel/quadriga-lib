# Questions

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/faq.html

| Section | Topics |
| --- | --- |
| [About the Library](#about-the-library) | [How is Quadriga-Lib related to QuaDRiGa?](#how-is-quadriga-lib-related-to-quadriga) · [How do I cite Quadriga-Lib?](#cite) · [Is the API stable?](#is-the-api-stable) · [Which compilers are supported?](#which-compilers-are-supported) · [MATLAB reports a "GLIBCXX ... not found" error when loading the MEX files. What can I do?](#matlab-reports-a-glibcxx-not-found-error-when-loading-the-mex-files-what-can-i-do) · [Can Quadriga-Lib be used together with other C++ libraries?](#can-quadriga-lib-be-used-together-with-other-c-libraries) |
| [Bug Reports and Contact](#bug-reports-and-contact) | [How do I report a bug?](#how-do-i-report-a-bug) · [How do I request a feature?](#how-do-i-request-a-feature) · [Where can I ask how to use the library?](#where-can-i-ask-how-to-use-the-library) · [How do I report a security problem?](#how-do-i-report-a-security-problem) · [How can I contribute code?](#how-can-i-contribute-code) · [Who develops Quadriga-Lib?](#who-develops-quadriga-lib) · [How can I contact the developers?](#how-can-i-contact-the-developers) |
| [Distribution License](#distribution-license) | [What is the distribution license for Quadriga-Lib?](#what-is-the-distribution-license-for-quadriga-lib) · [Can I use Quadriga-Lib in commercial products without releasing the source code?](#can-i-use-quadriga-lib-in-commercial-products-without-releasing-the-source-code) · [How does the Apache License compare to the GPL, MIT and BSD licenses?](#how-does-the-apache-license-compare-to-the-gpl-mit-and-bsd-licenses) · [Is Quadriga-Lib a product of the Apache Software Foundation?](#is-quadriga-lib-a-product-of-the-apache-software-foundation) |
| [Supporting the Project](#supporting-the-project) | [How can I support the development?](#how-can-i-support-the-development) · [Other ways to help](#other-ways-to-help) |

<a id="about-the-library"></a><a id="e63d3e38"></a>
## About the Library

<a id="how-is-quadriga-lib-related-to-quadriga"></a>
### How is Quadriga-Lib related to QuaDRiGa?

[QuaDRiGa](https://quadriga-channel-model.de) is a geometry-based stochastic channel model written in MATLAB. Quadriga-Lib is its C++ successor by the same lead developer: the model components are being rewritten in C++ for speed, with thin interfaces to MATLAB, Octave and Python. It is not a drop-in replacement for the MATLAB toolbox; functions, names and file formats follow their own API. [QuaDRiGa-NG](https://github.com/stjaeckel/QuaDRiGa-NG) is a fork of the MATLAB QuaDRiGa toolbox that uses Quadriga-Lib in its backend.

<a id="cite"></a>
### How do I cite Quadriga-Lib?

If you use Quadriga-Lib in published work, please cite the software and the version you used. The installed version is returned by `quadriga_lib.version()` in Python, `quadriga_lib.version` in MATLAB / Octave and `quadriga_lib::quadriga_lib_version()` in C++. Publications on the models are listed on the [Publications](publications.md) page.

```
@software{quadriga_lib,
  author  = {Jaeckel, Stephan},
  title   = {Quadriga-Lib: C++/MATLAB/Python library for radio channel modelling and simulation},
  url     = {https://github.com/stjaeckel/quadriga-lib},
  version = {<version>},
  year    = {<year>}
}
```

<a id="is-the-api-stable"></a>
### Is the API stable?

Quadriga-Lib is under active development. Breaking changes are avoided where possible, but they do happen between releases. For reproducible results, use a release version and note it, or check out a release tag of the [GitHub repository](https://github.com/stjaeckel/quadriga-lib). The `main` branch may be ahead of the latest release. The API documentation on this site is generated from the source code of the latest release.

<a id="which-compilers-are-supported"></a>
### Which compilers are supported?

Quadriga-Lib needs C++17. It is built and tested with GCC 11 or newer (GCC 12 by default on Linux) and with MSVC 2022 on Windows.

<a id="matlab-reports-a-glibcxx-not-found-error-when-loading-the-mex-files-what-can-i-do"></a>
### MATLAB reports a "GLIBCXX ... not found" error when loading the MEX files. What can I do?

MATLAB ships its own, older C++ runtime (`libstdc++`). MEX files built with a newer GCC than your MATLAB release supports fail to load with this error. The build uses GCC 12 by default. If your MATLAB release needs a different version, install it (e.g. `sudo apt install g++-11`) and build with `make gcc_version=11`. The compilers supported by each MATLAB release are listed by MathWorks.

<a id="can-quadriga-lib-be-used-together-with-other-c-libraries"></a>
### Can Quadriga-Lib be used together with other C++ libraries?

Yes. Quadriga-Lib uses [Armadillo](https://arma.sourceforge.net) for its matrices and cubes. To exchange data with other libraries, create Armadillo objects that use auxiliary memory, use their STL-style iterators, or get a pointer to the matrix memory with `.memptr()`.

<a id="bug-reports-and-contact"></a><a id="59d38498"></a>
## Bug Reports and Contact

<a id="how-do-i-report-a-bug"></a>
### How do I report a bug?

Open an issue with the [bug report form](https://github.com/stjaeckel/quadriga-lib/issues/new?template=bug_report.yml) on GitHub. The most useful part is a small, self-contained example that shows the problem, together with the library version and the interface (C++, Python, MATLAB or Octave). Please search the [existing issues](https://github.com/stjaeckel/quadriga-lib/issues?q=is%3Aissue) first; if your problem is already there, add your case as a comment.

<a id="how-do-i-request-a-feature"></a>
### How do I request a feature?

Use the [feature request form](https://github.com/stjaeckel/quadriga-lib/issues/new?template=feature_request.yml). Describe the task you want to solve, not only the function you have in mind.

<a id="where-can-i-ask-how-to-use-the-library"></a>
### Where can I ask how to use the library?

Ask in [GitHub Discussions](https://github.com/stjaeckel/quadriga-lib/discussions). You can also share your results there.

<a id="how-do-i-report-a-security-problem"></a>
### How do I report a security problem?

Please do not open a public issue. Use GitHub's [private vulnerability reporting](https://github.com/stjaeckel/quadriga-lib/security/advisories/new) instead; only the maintainer sees the report.

<a id="how-can-i-contribute-code"></a>
### How can I contribute code?

Small fixes are welcome as pull requests on [GitHub](https://github.com/stjaeckel/quadriga-lib). For anything larger, please open an issue first so the approach can be agreed before you write code. Contributions need tests and documentation; see [CONTRIBUTING.md](https://github.com/stjaeckel/quadriga-lib/blob/main/CONTRIBUTING.md).

<a id="who-develops-quadriga-lib"></a>
### Who develops Quadriga-Lib?

Lead development is done by [Stephan Jaeckel](https://sjc-wireless.com).

<a id="how-can-i-contact-the-developers"></a>
### How can I contact the developers?

By email to "info \[ÄT] quadriga-lib \[DÖT] org". For bugs, feature requests and usage questions, GitHub is preferred, so that other users can find the answers.

<a id="distribution-license"></a><a id="89373af1"></a>
## Distribution License

<a id="what-is-the-distribution-license-for-quadriga-lib"></a>
### What is the distribution license for Quadriga-Lib?

Quadriga-Lib is licensed under the [Apache License 2.0](https://opensource.org/license/apache-2-0), a permissive license. See also a [quick summary](https://www.tldrlegal.com/license/apache-license-2-0-apache-2-0) of the license.

<a id="can-i-use-quadriga-lib-in-commercial-products-without-releasing-the-source-code"></a>
### Can I use Quadriga-Lib in commercial products without releasing the source code?

Yes. Quadriga-Lib can be used in open-source and proprietary software. If you distribute it, in source or binary form, keep its copyright and license notice (the `LICENSE` file) with your software or its documentation.

<a id="how-does-the-apache-license-compare-to-the-gpl-mit-and-bsd-licenses"></a>
### How does the Apache License compare to the GPL, MIT and BSD licenses?

Unlike the GPL, the Apache License does not require you to publish the source code of software that uses Quadriga-Lib. It is a permissive license like MIT and BSD, and it also grants users the patent rights of the contributors.

<a id="is-quadriga-lib-a-product-of-the-apache-software-foundation"></a>
### Is Quadriga-Lib a product of the Apache Software Foundation?

No. It only uses the Apache License.

<a id="supporting-the-project"></a><a id="d8094d72"></a>
## Supporting the Project

<a id="how-can-i-support-the-development"></a>
### How can I support the development?

If Quadriga-Lib is useful to you, you can support its development with a donation. It lets me spend more time on new features, bug fixes and documentation.

- [GitHub Sponsors](https://github.com/sponsors/stjaeckel)
- [PayPal](https://www.paypal.com/donate/?hosted_button_id=B2N5HS5F4HHFE)

<a id="other-ways-to-help"></a>
### Other ways to help

Report bugs with a small example, answer questions in [Discussions](https://github.com/stjaeckel/quadriga-lib/discussions), and cite the library in your publications.
