# qrt_file_parse
Read metadata from a QRT file

## Usage:
```
from quadriga_lib import channel

# Separate outputs
no_cir, no_orig, no_dest, no_freq, cir_offset, orig_names, dest_names, version, fGHz, cir_pos, cir_orientation, orig_pos, orig_orientation = channel.qrt_file_parse( fn )

# Output as tuple
data = channel.qrt_file_parse( fn )
```

## Input Argument:
- **`fn`**<br>
  Filename of the QRT file, string

## Output Arguments:
- **`no_cir`**<br>
  Number of channel snapshots per origin point

- **`no_orig`**<br>
  Number of origin points (e.g., TXs)

- **`no_dest`**<br>
  Number of destinations (RX)

- **`no_freq`**<br>
  Number of frequencies

- **`cir_offset`**<br>
  CIR offset for each destination

- **`orig_names`**<br>
  Names of the origin points (TXs), list of strings

- **`dest_names`**<br>
  Names of the destination points (RXs), list of strings

- **`version`**<br>
  QRT file version

- **`fGHz`**<br>
  Center frequency in GHz as stored in the QRT file, numpy array of floats

- **`cir_pos`**<br>
  CIR positions in Cartesian coordinates, numpy array of shape [no_cir, 3]

- **`cir_orientation`**<br>
  CIR orientation in Euler angles in rad, numpy array of shape [no_cir, 3]

- **`orig_pos`**<br>
  Origin (TX) positions in Cartesian coordinates, numpy array of shape [no_orig, 3]

- **`orig_orientation`**<br>
  Origin (TX) orientations in Euler angles in rad, numpy array of shape [no_orig, 3]
  
  
  ---
  # qrt_file_read
Read ray-tracing data from QRT file

## Usage:
```
from quadriga_lib import channel
data = channel.qrt_file_read( fn, i_cir, i_orig, downlink )
```

## Input Arguments:
- **`fn`**<br>
  Filename of the QRT file, string

- **`cir`**<br>
  Snapshot index in the file, Default = 0

- **`orig`**<br>
  Origin index (for downlink Origin = TX), Default = 0

- **`downlink`**<br>
  Switch for uplink / downlink direction, Default = true (downlink)

- **`normalize_M`**<br>
   Switch for different normalization options:
   0 | M as in QRT file, path_gain as FSPL (no normalization)
   1 | M has sum-column power is 2, path_gain is FSPL + material losses (default)

## Output Arguments:
- **`data`**<br>
  Dictionary containing the data in the HDF file with the following keys:
  `center_freq`    | Center frequency in [Hz]                                 | Length `[n_freq]`
  `tx_pos`         | Transmitter position                                     | Length `[3]`
  `tx_orientation` | Transmitter orientation, Euler angles, rad               | Length `[3]`
  `rx_pos`         | Receiver position                                        | Length `[3]`
  `rx_orientation` | Receiver orientation, Euler angles, rad                  | Length `[3]`
  `fbs_pos`        | First-bounce scatterer positions                         | Size `[3, n_path]`
  `lbs_pos`        | Last-bounce scatterer positions                          | Size `[3, n_path]`
  `path_gain`      | Path gain before antenna, linear scale                   | Size `[n_path, n_freq]`
  `path_length`    | Path length from TX to RX phase center in m              | Length `[n_path]`
  `M`              | Polarization transfer function, interleaved complex      | Size `[8, n_path, n_freq]` or `[2, n_path, n_freq]`
  `aod`            | Departure azimuth angles in [rad]                        | Length `[n_path]`
  `eod`            | Departure elevation angles in [rad]                      | Length `[n_path]`
  `aoa`            | Arrival azimuth angles in [rad]                          | Length `[n_path]`
  `eoa`            | Arrival elevation angles in [rad]                        | Length `[n_path]`
  `path_coord`     | Interaction coordinates                                  | List of `[3, n_int_s]`
  
  
  ---
  # calc_delay_spread
Calculate the RMS delay spread in [s]

## Description:
- Computes the root-mean-square (RMS) delay spread from a given set of delays and corresponding
  linear-scale powers for each channel impulse response (CIR).
- An optional power threshold in [dB] relative to the strongest path can be applied. Paths with
  power below `p_max(dB) - threshold` are excluded from the calculation.
- An optional granularity parameter in [s] groups paths in the delay domain. Powers of paths
  falling into the same delay bin are summed before computing the delay spread.
- Optionally returns the mean delay for each CIR.

## Usage:
```
import quadriga_lib
ds, mean_delay = quadriga_lib.tools.calc_delay_spread(delays, powers, threshold=100.0, granularity=0.0)
```

## Arguments:
- `list of np.ndarray **delays**` (input)<br>
  Delays in [s]. A list of length `n_cir`, where each element is a 1D numpy array of path delays.

- `list of np.ndarray **powers**` (input)<br>
  Path powers on a linear scale [W]. Same structure as `delays`.

- `float **threshold** = 100.0` (input)<br>
  Power threshold in [dB] relative to the strongest path. Default: 100 dB.

- `float **granularity** = 0.0` (input)<br>
  Window size in [s] for grouping paths in the delay domain. Default: 0 (no grouping).

## Returns:
- `np.ndarray **ds**` (output)<br>
  RMS delay spread in [s] for each CIR. Shape `(n_cir,)`.

- `np.ndarray **mean_delay**` (output)<br>
  Mean delay in [s] for each CIR. Shape `(n_cir,)`.

## Example:
```
import numpy as np
import quadriga_lib

delays = [np.array([0.0, 1e-6, 2e-6])]
powers = [np.array([1.0, 0.5, 0.25])]
ds, mean_delay = quadriga_lib.calc_delay_spread(delays, powers)
```

---
# calc_rician_k_factor
Calculate the Rician K-Factor from channel impulse response data

## Description:
- The Rician K-Factor (KF) is defined as the ratio of signal power in the dominant line-of-sight
  (LOS) path to the power in the scattered (non-line-of-sight, NLOS) paths.
- The LOS path is identified by matching the absolute path length with the direct distance between
  TX and RX positions (`dTR`).
- All paths arriving within `dTR + window_size` are considered LOS and their power is summed.
- Paths arriving after `dTR + window_size` are considered NLOS and their power is summed.
- If the total NLOS power is zero, the K-Factor is set to infinity.
- If the total LOS power is zero, the K-Factor is set to zero.
- The transmitter and receiver positions can be fixed (shape `(3,)` or `(3, 1)`) or mobile
  (shape `(3, n_cir)`). Fixed positions are reused for all channel snapshots.
- Output `pg` returns the total path gain (sum of all path powers) for each snapshot.

## Usage:
```
import quadriga_lib
kf, pg = quadriga_lib.tools.calc_rician_k_factor( powers, path_length, tx_pos, rx_pos, window_size=0.01 )
```

## Arguments:
- `list of ndarray **powers**` (input)<br>
  Path powers in Watts [W]. List of length `n_cir`, where each element is a 1D numpy array of
  length `n_path`.

- `list of ndarray **path_length**` (input)<br>
  Absolute path lengths from TX to RX phase center in meters. List of length `n_cir`, where each
  element is a 1D numpy array of length `n_path` matching the corresponding entry in `powers`.

- `ndarray **tx_pos**` (input)<br>
  Transmitter position in Cartesian coordinates. Shape `(3,)` or `(3, 1)` for fixed TX, or
  `(3, n_cir)` for mobile TX.

- `ndarray **rx_pos**` (input)<br>
  Receiver position in Cartesian coordinates. Shape `(3,)` or `(3, 1)` for fixed RX, or
  `(3, n_cir)` for mobile RX.

- `float **window_size** = 0.01` (input)<br>
  LOS window size in meters. Paths with length ≤ `dTR + window_size` are considered LOS.

## Returns:
- `ndarray **kf**` (output)<br>
  Rician K-Factor on linear scale. Shape `(n_cir,)`.

- `ndarray **pg**` (output)<br>
  Total path gain (sum of path powers). Shape `(n_cir,)`.
  
  
---
# calc_angular_spreads_sphere
Calculate azimuth and elevation angular spreads with spherical wrapping

## Description:
- Calculates the RMS azimuth and elevation angular spreads from a set of power-weighted angles.
- Inputs use lists of 1D numpy arrays so that each CIR can have a different number of paths.
- Uses spherical coordinate wrapping to avoid the pole singularity: the power-weighted mean
  direction is computed in Cartesian coordinates and all paths are rotated so the centroid lies
  on the equator before computing spreads.
- Without spherical wrapping, azimuth spread near the poles is inflated (large azimuth spread
  despite energy being focused into a small solid angle). This method corrects for that.
- Optionally computes an optimal bank (roll) angle that maximizes azimuth spread and minimizes
  elevation spread, corresponding to the principal axes of the angular power distribution.
- Setting `disable_wrapping` to True skips the rotation and computes spreads directly from
  raw angles.

## Usage:
```
import quadriga_lib
as_spread, es_spread, orientation, phi, theta = quadriga_lib.tools.calc_angular_spreads_sphere(
    az, el, powers)
as_spread, es_spread, orientation, phi, theta = quadriga_lib.tools.calc_angular_spreads_sphere(
    az, el, powers, disable_wrapping=False, calc_bank_angle=True, quantize=0.0)
```

## Arguments:
- `list of np.ndarray **az**` (input)<br>
  Azimuth angles in [rad]. List of length `n_cir`, each element is a 1D array of length `n_path`.

- `list of np.ndarray **el**` (input)<br>
  Elevation angles in [rad]. List of length `n_cir`, each element is a 1D array of length `n_path`.

- `list of np.ndarray **powers**` (input)<br>
  Path powers in [W]. List of length `n_cir`, each element is a 1D array of length `n_path`.

- `bool **disable_wrapping** = False` (input)<br>
  If True, skip spherical rotation and compute spreads from raw angles.

- `bool **calc_bank_angle** = True` (input)<br>
  If True, compute the optimal bank angle analytically.

- `float **quantize** = 0.0` (input)<br>
  Angular quantization step in [deg]. Set to 0 for no quantization.

## Returns:
- `np.ndarray **azimuth_spread**` (output)<br>
  RMS azimuth angular spread in [rad]. Shape `(n_cir,)`.

- `np.ndarray **elevation_spread**` (output)<br>
  RMS elevation angular spread in [rad]. Shape `(n_cir,)`.

- `np.ndarray **orientation**` (output)<br>
  Mean-angle orientation [bank, tilt, heading] in [rad]. Shape `(3, n_cir)`.

- `list of np.ndarray **phi**` (output)<br>
  Rotated azimuth angles in [rad]. List of length `n_cir`.

- `list of np.ndarray **theta**` (output)<br>
  Rotated elevation angles in [rad]. List of length `n_cir`.

## Example:
```
import numpy as np
import quadriga_lib

az = [np.array([0.1, -0.1, 0.05]), np.array([0.2, -0.2, 0.1, -0.1])]
el = [np.array([0.0, 0.0, 0.0]), np.array([0.05, -0.05, 0.0, 0.0])]
powers = [np.array([1.0, 1.0, 0.5]), np.array([2.0, 1.0, 1.5, 0.5])]
as_spread, es_spread, orient, phi, theta = quadriga_lib.tools.calc_angular_spreads_sphere(
    az, el, powers)
```

---
# calc_cross_polarization_ratio
Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases

## Description:
- Computes the aggregate cross-polarization ratio (XPR) from the polarization transfer matrices
  of all channel impulse responses (CIRs) using the total-power-ratio method.
- For each CIR, the total co-polarized and cross-polarized received powers are accumulated
  across all qualifying paths, and the XPR is obtained as a single ratio of the totals.
- In addition to the linear V/H basis, the XPR is also computed in the circular LHCP/RHCP basis.
- The LOS path is identified by comparing each path's absolute length against the direct
  TX-RX distance. All paths with `path_length < dTR + window_size` are excluded from
  the XPR calculation by default (controlled by `include_los`).
- If the total cross-polarized power is zero, the XPR is set to 0 (undefined).

## Usage:
```
import quadriga_lib
xpr, pg = quadriga_lib.tools.calc_cross_polarization_ratio( powers, M, path_length, tx_pos, rx_pos )
xpr, pg = quadriga_lib.tools.calc_cross_polarization_ratio( powers, M, path_length, tx_pos, rx_pos, include_los, window_size )
```

## Arguments:
- `list of np.ndarray **powers**` (input)<br>
  Path powers in Watts. List of length `n_cir`, each element is a 1D array of length `n_path`.

- `list of np.ndarray **M**` (input)<br>
  Polarization transfer matrices. List of length `n_cir`, each element is a 2D array of size `[8, n_path]`.

- `list of np.ndarray **path_length**` (input)<br>
  Absolute path length from TX to RX in meters. List of length `n_cir`, each element is a 1D array of length `n_path`.

- `np.ndarray **tx_pos**` (input)<br>
  Transmitter position. Size `[3, 1]` (fixed) or `[3, n_cir]` (mobile).

- `np.ndarray **rx_pos**` (input)<br>
  Receiver position. Size `[3, 1]` (fixed) or `[3, n_cir]` (mobile).

- `bool **include_los** = False` (input)<br>
  If `True`, include LOS paths in the XPR calculation.

- `float **window_size** = 0.01` (input)<br>
  LOS window size in meters.

## Returns:
- `np.ndarray **xpr**` (output)<br>
  Cross-polarization ratio, linear scale. Size `[n_cir, 6]`.
  Columns: 0=aggregate linear, 1=V-XPR, 2=H-XPR, 3=aggregate circular, 4=LHCP, 5=RHCP.

- `np.ndarray **pg**` (output)<br>
  Total path gain over all paths. 1D array of length `[n_cir]`.


---
# HDF5_CREATE_FILE
Create a new HDF5 channel file with a custom storage layout

## Description:
Quadriga-Lib offers an HDF5-based method for storing and managing channel data. A key feature of this
library is its ability to organize multiple channels within a single HDF5 file while enabling access
to individual data sets without the need to read the entire file. In this system, channels can be
structured in a multi-dimensional array. For instance, the first dimension might represent the Base
Station (BS), the second the User Equipment (UE), and the third the frequency. However, it is important
to note that the dimensions of the storage layout must be defined when the file is initially created
and cannot be altered thereafter. The function `quadriga_lib.channel.hdf5_create_file` is used to create an
empty file with a predetermined custom storage layout.

## Usage:
```
from quadriga_lib import channel
channel.hdf5_create_file( fn, nx, ny, nz, nw )
```

## Input Arguments:
- **`fn`**<br>
  Filename of the HDF5 file, string

- **`nx`** (optional)<br>
  Number of elements on the x-dimension, Default = 65536

- **`ny`** (optional)<br>
  Number of elements on the x-dimension, Default = 1

- **`nz`** (optional)<br>
  Number of elements on the x-dimension, Default = 1

- **`nw`** (optional)<br>
  Number of elements on the x-dimension, Default = 1
  
  
---
# HDF5_WRITE_DSET
Writes unstructured data to a HDF5 file

## Description:
Quadriga-Lib offers a solution based on HDF5 for storing and organizing channel data. In addition
to structured datasets, the library facilitates the inclusion of extra datasets of various types
and shapes. This feature is particularly beneficial for integrating descriptive data or analysis
results. The function `quadriga_lib.channel.hdf5_write_dset` writes a single unstructured dataset.

## Usage:

```
from quadriga_lib import channel
channel.hdf5_write_dset( fn, ix, iy, iz, iw, name, data );
```

## Input Arguments:
- **`fn`**<br>
  Filename of the HDF5 file, string

- **`ix`**<br>
  Storage index for x-dimension, Default = 0

- **`iy`**<br>
  Storage index for y-dimension, Default = 0

- **`iz`**<br>
  Storage index for z-dimension, Default = 0

- **`iw`**<br>
  Storage index for w-dimension, Default = 0

- **`name`**<br>
  Name of the dataset; String

- **`data`**<br>
  Data to be written

## Caveat:
- Throws an error if dataset already exists at this location
- Throws an error if file does not exist (use hdf5_create_file)
- Supported types: string, double, float, (u)int32, (u)int64
- Supported size: up to 3 dimensions
- Storage order is maintained




