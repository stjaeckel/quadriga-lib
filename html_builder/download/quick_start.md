/*!SECTION
Quick Start
SECTION!*/

/*!SECTION_DESC

<br><br>

All three examples create a 2 × 2 3GPP antenna array at 3.7 GHz and print the directivity of the first element (about 9.8 dBi). More examples are in the API references.

<br><br>

**Python**<br>
```
import quadriga_lib

ant = quadriga_lib.arrayant.generate('3gpp', M=2, N=2, freq=3.7e9)
print(quadriga_lib.version())
print(quadriga_lib.arrayant.calc_directivity(ant, 0))
```

<br>

**MATLAB / Octave**<br>
```
ant = quadriga_lib.arrayant_generate('3GPP', 1, 3.7e9, [], [], [], 2, 2);
disp(quadriga_lib.version())
disp(quadriga_lib.arrayant_calc_directivity(ant, 1))
```

<br>

**C++**<br>
```
#include <quadriga_lib.hpp>
#include <iostream>

int main()
{
    auto ant = quadriga_lib::generate_arrayant_3GPP<double>(2, 2, 3.7e9);
    std::cout << quadriga_lib::quadriga_lib_version() << "\n";
    std::cout << ant.calc_directivity_dBi(0) << " dBi\n";
}
```
Compile in the source folder after `make`:
```
g++ -std=c++17 -O2 -fopenmp -I include -I external/armadillo-14.2.2/include example.cpp lib/libquadriga.a -o example
```

SECTION_DESC!*/
