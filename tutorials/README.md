# Tutorials

Tutorial notebooks, `<stem>.ipynb`, published on the website by the website builder. They are committed cleared: no outputs, no execution counts.

## Running them

Run the notebooks in an environment with quadriga-lib and jupyter (the set-up is described on the tutorial page of the website):

```bash
make -C tutorials
```

or `make tutorials` from the repository root. Four notebooks run in parallel; `JOBS=8` or `JOBS=1` changes that. Each notebook is run into `executed/<stem>.ipynb` (ignored by git), with its outputs; the notebook itself is not changed. A notebook runs again only when it is newer than its executed copy; after a change of quadriga-lib or of the environment, run `make -C tutorials clean` first. Two runs in the same environment give byte-identical copies. `PYTHON=...` selects another Python, e.g. `make -C tutorials PYTHON=/path/to/env/bin/python`.

A failing cell stops the run of that notebook with the error and leaves no executed copy.

## Clearing them

```bash
make -C tutorials clean
```

deletes `executed/` and clears the notebooks in place: outputs, execution counts and the metadata a run adds are removed; cell ids and the tutorial metadata (`quadriga_tutorial`, with the alt texts of the figures) stay. Clear a notebook before committing it if it was run in place, e.g. in JupyterLab. The root `make clean` keeps `executed/`; the root `make tidy` runs this clean.

## Files

| Path | Content |
|------|---------|
| `<stem>.ipynb` | The tutorials, cleared |
| `executed/` | The executed copies (not in git) |
| `GNUmakefile` | Run and clean targets |
| `qnb/run.py` | Runs one notebook into its executed copy (nbclient, no execution timestamps, temporary file renamed on success) |
| `qnb/clear.py` | Clears notebooks in place (standard library only) |
