/*!SECTION
Installation
SECTION!*/

/*!SECTION_DESC

<br><br>

**Linux / Ubuntu**<br>

- [**OPTIONAL**]: Install Matlab from <a href="https://www.mathworks.com">www.mathworks.com</a> - you need to obtain a license
- Install the required packages (octave-dev is only needed if you want to use Octave):
```
sudo apt install bzip2 gcc git make cmake g++ libhdf5-dev python3-dev python3-pytest python3-numpy octave-dev python3-pip python3-venv octave-dev
```
- Get Quadriga-Lib either from Github or <a href="http://quadriga-lib.org">quadriga-lib.org</a> 
```
git clone https://github.com/stjaeckel/quadriga-lib
cd quadriga-lib
```
- Compile Quadriga-Lib using `make` (cmake is used internally and can also be used directly)
- [**OPTIONAL**]: Build and run tests: `make test`
- [**OPTIONAL**]: Add the library to your MATLAB path by running `addpath('quadriga-lib')` in the MATLAB command window or by adding it to your `startup.m` file.
- [**OPTIONAL**]: If you want to use the Python API, add the library to your Python path by adding the library path to your `PYTHONPATH` environment variable. You can do this by adding the following line to your `.bashrc` or `.bash_profile` file:
```
export PYTHONPATH=$PYTHONPATH:/path/to/quadriga-lib/lib
```
<br><br>

**Linux / Ubuntu with Anaconda**<br>

- Install Miniconda from [Anaconda's official website](https://www.anaconda.com/products/distribution#download-section). If you already have Anaconda installed, you can skip this step.
```
cd ~/Downloads
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
- Setup a new Anaconda environment with the required packages (you can also use the default channel instead of `conda-forge`, but octave is only available in `conda-forge`):
```
conda deactivate
conda create --name quadriga-lib -c conda-forge python=3.13 numpy pandas jupyterlab seaborn pytest scipy jupyterlab_widgets ipywidgets traittypes jupyter compilers make cmake hdf5 octave
conda activate quadriga-lib
```
- Get Quadriga-Lib either from Github or [quadriga-lib.org](http://quadriga-lib.org)
```
git clone https://github.com/stjaeckel/quadriga-lib
cd quadriga-lib
```
- Compile Quadriga-Lib (C++, MATLAB, Octave, Python) using `make` (cmake is used internally and can also be used directly)
- [**OPTIONAL**]: Build and run tests: `make test`
- [**ALTERNATIVE**]: If you only want to use the Python API, use `make python` and `make python_test`.
- [**OPTIONAL**]: Add the library to your MATLAB path by running `addpath('quadriga-lib')` in the MATLAB command window or by adding it to your `startup.m` file.
- [**OPTIONAL**]: If you want to use the Python API, install the Python package by running: `make python_install`. This will install the `quadriga-lib` package into your Anaconda environment.

<br><br>

**Windows**<br>
- Octave and Python are not supported yet (only MATLAB is)
- Install Build Tools for Visual Studio 2022 from https://visualstudio.microsoft.com
- Install Matlab from (https://www.mathworks.com) - you need to obtain a licence
- From MATLAB Shell run "mex -setup -v" and select compiler MSVC Compiler
- If error: https://yingzhouli.com/posts/2020-06/mex-msvc.html
- Open "x64 Native Tools Command Prompt" from start menu
- Navigate to library path, e.g. `cd Z:\quadriga-lib`
- Run `nmake` to compile `quadriga-lib` and the MEX Matlab interface
- [**OPTIONAL**]: Compile and run the unit tests by running `nmake test`
- [**OPTIONAL**]: Add the library to your MATLAB path by running `addpath('quadriga-lib')` in the MATLAB command window or by adding it to your `startup.m` file.

SECTION_DESC!*/