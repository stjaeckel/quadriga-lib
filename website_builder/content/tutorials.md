+++
title = "Tutorials"
description = "Tutorials for Quadriga-Lib: Jupyter notebooks in Python and MATLAB that explain one topic each, step by step, with working code."
lead = "Each tutorial explains one topic step by step, with code and its results. Read it here, or download the notebook and run it yourself; the Setup section below shows what you need."
+++

## Setup

The Python tutorials need Python with Quadriga-Lib, numpy, matplotlib and Jupyter. With [Miniforge](https://conda-forge.org/download/) (conda with the conda-forge channel only):

```bash
conda create -n quadriga --override-channels -c conda-forge python=3.14 pip numpy
conda activate quadriga
pip install quadriga-lib
conda install --override-channels -c conda-forge matplotlib jupyter
jupyter lab
```

The first three lines set up the environment `quadriga` as on the [Get started](download.html) page; if you have it already, activate it and continue with the fourth line. `jupyter lab` opens JupyterLab in the browser; open a downloaded notebook there and run its cells. Each tutorial names the oldest Quadriga-Lib release it needs in its first paragraph.

The MATLAB variants need a Jupyter kernel in the same environment and the MATLAB/Octave interface of Quadriga-Lib on the path:

- Octave 8.4 or newer (Linux): `conda install --override-channels -c conda-forge octave octave_kernel`
- MATLAB R2024a or newer: `pip install jupyter-matlab-proxy` and `conda env config vars set MWI_USE_EXISTING_LICENSE=True`, so that the kernel uses the license of the installed MATLAB; activate the environment again afterwards.
- The interface: the Windows or Ubuntu package, or a build from source, and `addpath('/path/to/quadriga-lib')` in `startup.m` (MATLAB) or `~/.octaverc` (Octave, also read by its kernel), as described for [Windows](download.html#windows), [Ubuntu](download.html#ubuntu) and [Linux with conda](download.html#linux-conda) on the Get started page.

A MATLAB variant runs with the kernel Octave or MATLAB Kernel (in JupyterLab: Kernel, Change Kernel); the MATLAB kernel starts MATLAB in the background, which takes about half a minute at the first cell.
