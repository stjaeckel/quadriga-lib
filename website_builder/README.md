# Website builder

## Set up a build machine

The website builder (`make website`) needs Python 3.10 or newer with markdown-it-py, mdit-py-plugins, Pygments and Jinja2, and pandoc 3.7 or newer on `PATH`; the tests also need pytest. Set this up once per machine, in one of two ways.

1. Conda (any Linux; the documented path):

```bash
conda create -y -n quadriga-web -c conda-forge "python>=3.11" "markdown-it-py>=3.0" "mdit-py-plugins>=0.4.2" pygments "jinja2>=3.1" pytest "pandoc>=3.7"
conda activate quadriga-web
```

2. Ubuntu 26.04 or newer (older releases ship an older pandoc; use conda there):

```bash
sudo apt install pandoc python3-markdown-it python3-mdit-py-plugins python3-pygments python3-jinja2 python3-pytest
```

With conda, activate the environment before each build session. To check the set-up, run `make website`: if anything is missing, it stops with a plain message that lists every missing item. Upgrade with `conda update -n quadriga-web --all` or with the system's package updates. Nothing is installed into the working copy.
