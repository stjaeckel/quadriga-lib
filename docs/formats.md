# Data formats

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/formats.html

| Format | Summary |
| --- | --- |
| [QuaDRiGa Array Antenna Exchange Format (QDANT)](formats/qdant.md) | XML file format for antenna array patterns: elements, positions, port weights and the layout of several antennas |
| [Material Properties](formats/materials.md) | The material table of the ray functions: CSV columns, built-in EM materials and the .mtl mapping |
| [Layout CSV files](formats/layout_csv.md) | Two CSV files that hold a network layout: the Tx and Rx nodes with their tracks, scenarios and antennas |
| [HDF5 channel files](formats/hdf5_channel.md) | HDF5 file of channels with antennas: a grid of storage places, one channel with coefficients, delays, path data and metadata in each |
| [QRT path files](formats/qrt.md) | Binary file of propagation paths without antennas, from ray tracing or the GBSM: one block of paths per origin (Tx) for a list of CIR positions |
