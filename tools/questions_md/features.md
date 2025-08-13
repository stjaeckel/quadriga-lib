/*!SECTION
Features / Functions
SECTION!*/

/*!SECTION_DESC
- **Is the API stable?**<br>
  Quadriga-Lib is under active development, and the API may change in future releases.
  However, we try to keep the API stable and avoid breaking changes. If you use the
  API in your code, we recommend to use the latest release version of Quadriga-Lib
  to ensure compatibility. If you need to use a specific version of the API, you can
  use the `git checkout` command to switch to a specific commit or tag in the repository.
  The API documentation is generated from the source code, so it will always match the
  current state of the code. 

- **What is the minimum required C++ version?**<br>
  Quadriga-Lib requires C++17. The code is tested with GCC 11 and later, and with
  MSVC 14. You may need to enable C++17 support explicitly (e.g., by using 
  the `-std=c++17` flag).

- **Is it possible to interface Quadriga-Lib with other libraries ?**<br>
  Yes, Quadriga-Lib is designed to be easily integrated with other libraries. 
  It uses <a href="https://arma.sourceforge.net">Armadillo</a> for linear algebra 
  operations, which is a C++ library that provides a high-level interface for matrix operations. 
  You can use Armadillo functions in your code, and Quadriga-Lib will work seamlessly with them. 
  Interfacing with other libraries can be done by creating matrices (or cubes) that use auxiliary 
  memory, or by accessing elements through STL-style iterators, or by directly obtaining a pointer 
  to matrix memory via the .memptr() function. 

SECTION_DESC!*/