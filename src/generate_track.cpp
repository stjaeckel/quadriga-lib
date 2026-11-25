// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_layout.hpp"
#include "qd_helpers_layout.hpp"
#include "qd_helpers_shared.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

/*!SECTION
Network layouts and mobility
SECTION!*/

namespace
{
constexpr double two_pi = 6.283185307179586;
constexpr double pi = 3.141592653589793;

// Angle wrapped to [-pi, pi]
inline double wrap_pi(double a) { return a - two_pi * std::round(a / two_pi); }

// Rotation matrix (column-major) of (bank, tilt, heading); and the Euler angles of a rotation matrix. At a tilt of
// +-90 deg, bank and heading are not unique; the bank is then 0
inline void euler_to_rot(double bank, double tilt, double heading, double *R)
{
    const double o[3] = {bank, tilt, heading};
    qd_rotation_matrix(o, R);
}

inline void rot_to_euler(const double *R, double &bank, double &tilt, double &heading)
{
    tilt = std::asin(std::max(-1.0, std::min(1.0, -R[2])));
    if (std::hypot(R[0], R[1]) > 1.0e-7)
    {
        heading = std::atan2(R[1], R[0]);
        bank = std::atan2(R[5], R[8]);
    }
    else
    {
        heading = std::atan2(-R[3], R[4]);
        bank = 0.0;
    }
}

// C = A B for 3x3 matrices in column-major order
inline void mat_mul(const double *A, const double *B, double *C)
{
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            C[3 * j + i] = A[i] * B[3 * j] + A[3 + i] * B[3 * j + 1] + A[6 + i] * B[3 * j + 2];
}

// Heading and tilt (positive down) of a direction vector; the heading of a vertical vector is 0
inline void heading_tilt(double dx, double dy, double dz, double &heading, double &tilt)
{
    const double h = std::hypot(dx, dy);
    heading = (h < 1.0e-12 * std::max(1.0, std::abs(dz))) ? 0.0 : std::atan2(dy, dx);
    tilt = -std::atan2(dz, h);
}

// Orientation 'out' (bank, tilt, heading) of an antenna with the stored orientation 'o' that is mounted on a vehicle moving
// with (heading, tilt); 'out' may be 'o'
inline void mounted(double heading, double tilt, const double *o, double *out)
{
    double Rm[9], Ro[9], R[9];
    euler_to_rot(0.0, tilt, heading, Rm);
    euler_to_rot(o[0], o[1], o[2], Ro);
    mat_mul(Rm, Ro, R);
    rot_to_euler(R, out[0], out[1], out[2]);
}

// Interpolation of an angle between a (w = 0) and b (w = 1) along the shorter way; the tilt is not periodic (it lives in
// [-pi/2, pi/2]) and is interpolated linearly
inline double lerp_angle(double a, double b, double w, bool periodic = true)
{
    if (a == b || w == 0.0)
        return a;
    if (!periodic)
        return a + w * (b - a);
    const double x = a + w * wrap_pi(b - a);
    return std::abs(x) > pi ? wrap_pi(x) : x;
}

// Checks the sizes of the members that the functions of this file read; with 'all_offsets', also the offsets of all nodes
// (else those of the node 'i_node' and of the next one, which keeps a call for one node of many cheap)
void check_structure(const quadriga_lib::layout_nodes &nodes, bool all_offsets = true, arma::uword i_node = 0)
{
    const arma::uword nn = (arma::uword)nodes.name.size(), nr = nodes.pos.n_cols;
    const std::string bad = "Input 'nodes' is not valid: ";
    if (nodes.offset.n_elem != nn || (nodes.pos.n_rows != 3 && !nodes.pos.is_empty()) || nodes.orientation.n_cols != nr ||
        (nodes.orientation.n_rows != 3 && !nodes.orientation.is_empty()))
        throw std::invalid_argument(bad + "the members 'name', 'offset', 'pos' and 'orientation' do not fit.");
    if ((!nodes.direction.is_empty() && (nodes.direction.n_rows != 2 || nodes.direction.n_cols != nr)) ||
        (!nodes.speed.is_empty() && nodes.speed.n_elem != nr) || (!nodes.time.is_empty() && nodes.time.n_elem != nr) ||
        (!nodes.relative.is_empty() && nodes.relative.n_elem != nn))
        throw std::invalid_argument(bad + "the members 'direction', 'speed', 'time' and 'relative' do not fit.");
    if ((!nodes.scenario.empty() && nodes.scenario.size() != nr) || (!nodes.indoor.is_empty() && nodes.indoor.n_elem != nr) ||
        (!nodes.floor.is_empty() && nodes.floor.n_elem != nr) || (!nodes.d2d_in.is_empty() && nodes.d2d_in.n_elem != nr))
        throw std::invalid_argument(bad + "the members 'scenario', 'indoor', 'floor' and 'd2d_in' do not fit.");
    for (const auto &col : nodes.extra)
        if (col.size() != nr)
            throw std::invalid_argument(bad + "the members 'extra' do not fit.");
    for (arma::uword i = all_offsets ? 0 : i_node; i < (all_offsets ? nn : std::min(nn, i_node + 2)); ++i)
        if ((i == 0 && nodes.offset[0] != 0) || (i > 0 && nodes.offset[i] <= nodes.offset[i - 1]) || nodes.offset[i] >= nr)
            throw std::invalid_argument(bad + "the member 'offset' must start at the first row, increase and stay within the rows.");
}

// Appends one node with one row at 'start' (empty = origin) and the default values; 'name' empty: the next free name,
// i.e. the prefix that all names share before their number (e.g. "UT"), else "N", with the next number
void append_node(quadriga_lib::layout_nodes &N, const std::string &name_in, const arma::vec &start)
{
    if (!start.is_empty() && (start.n_elem != 3 || !start.is_finite()))
        throw std::invalid_argument("Input 'start' must be empty or [x, y, z] with finite values.");
    std::string name = name_in;
    if (name.empty())
    {
        std::string prefix;
        bool common = !N.name.empty();
        for (const std::string &nm : N.name)
        {
            size_t p = nm.size();
            while (p > 0 && std::isdigit((unsigned char)nm[p - 1]))
                --p;
            if (p == 0 || p == nm.size() || (!prefix.empty() && nm.compare(0, p, prefix) != 0) || (!prefix.empty() && p != prefix.size()))
            {
                common = false;
                break;
            }
            prefix = nm.substr(0, p);
        }
        if (!common)
            prefix = "N";
        unsigned long long n = 0;
        for (const std::string &nm : N.name)
            if (nm.size() > prefix.size() && nm.size() <= prefix.size() + 18 && nm.compare(0, prefix.size(), prefix) == 0 &&
                nm.find_first_not_of("0123456789", prefix.size()) == std::string::npos)
                n = std::max(n, std::stoull(nm.substr(prefix.size())));
        name = prefix + std::to_string(n + 1);
    }
    else if (std::find(N.name.begin(), N.name.end(), name) != N.name.end())
        throw std::invalid_argument("Input 'name' is the name of an existing node ('" + name + "').");
    quadriga_lib::layout_nodes one; // The rules of the names
    one.name = {name}, one.sector = {0}, one.i_antenna = {0}, one.offset = {0}, one.pos.zeros(3, 1), one.orientation.zeros(3, 1);
    const std::string msg = one.is_valid();
    if (!msg.empty())
        throw std::invalid_argument("Input 'name': " + msg);

    const arma::uword nn = N.n_node(), nr = N.n_row();
    const double nan = arma::datum::nan;
    N.name.push_back(name);
    N.sector.resize(nn + 1), N.sector[nn] = 0;
    N.i_antenna.resize(nn + 1), N.i_antenna[nn] = 0;
    N.offset.resize(nn + 1), N.offset[nn] = nr;
    N.pos.insert_cols(nr, start.is_empty() ? arma::vec(3, arma::fill::zeros) : start);
    N.orientation.insert_cols(nr, arma::vec(3, arma::fill::zeros));
    if (!N.direction.is_empty())
        N.direction.insert_cols(nr, arma::vec{nan, 0.0});
    auto grow = [](auto &v, auto x)
    {
        if (!v.is_empty())
            v.resize(v.n_elem + 1), v[v.n_elem - 1] = x;
    };
    grow(N.speed, 0.0), grow(N.time, nan), grow(N.indoor, -1), grow(N.floor, -1), grow(N.d2d_in, nan);
    grow(N.seed, (arma::sword)-1), grow(N.relative, 0u), grow(N.cell, 1u);
    if (!N.scenario.empty())
        N.scenario.push_back("");
    for (auto &col : N.extra)
        col.push_back("");
    if (!N.clutter.is_empty())
        N.clutter.insert_cols(nn, arma::vec(3, arma::fill::value(nan)));
    for (auto &kv : N.scenpar_override)
        grow(kv.second, nan);
}

// Start of the messages about one node
std::string who(const quadriga_lib::layout_nodes &nodes, arma::uword i)
{
    return "Node '" + nodes.name[i] + "', sector " + (i < nodes.sector.n_elem ? std::to_string(nodes.sector[i]) : std::string("0")) + ": ";
}

inline bool is_relative(const quadriga_lib::layout_nodes &nodes, arma::uword i_node)
{
    return !nodes.relative.is_empty() && nodes.relative[i_node] == 1;
}

// Heading and tilt (positive down) of the steps between the n rows from 'first'. A step of zero length takes the previous
// direction, at the start the first one that exists. A vertical step (no horizontal part) keeps the previous heading and
// has the tilt -90 deg (up) or +90 deg (down): the vehicle still points where it pointed, so a mounted antenna does not turn
// by an arbitrary heading when the track starts to climb. NG (cart2sph) gives such a step the heading 0; this is a
// deliberate deviation. Returns false if no step moves (the arrays are then 0)
bool step_directions(const arma::mat &pos, arma::uword first, arma::uword n, std::vector<double> &heading, std::vector<double> &tilt)
{
    const arma::uword ns = n - 1;
    heading.assign(ns, 0.0);
    tilt.assign(ns, 0.0);
    std::vector<char> moves(ns, 0), horizontal(ns, 0);
    for (arma::uword k = 0; k < ns; ++k)
    {
        const double dx = pos(0, first + k + 1) - pos(0, first + k), dy = pos(1, first + k + 1) - pos(1, first + k),
                     dz = pos(2, first + k + 1) - pos(2, first + k);
        if (dx == 0.0 && dy == 0.0 && dz == 0.0)
            continue;
        double h;
        heading_tilt(dx, dy, dz, h, tilt[k]);
        moves[k] = 1;
        if (std::hypot(dx, dy) >= 1.0e-12 * std::max(1.0, std::abs(dz)))
            heading[k] = h, horizontal[k] = 1;
    }
    // Steps without a value take the previous one, at the start the first one that has it
    auto fill = [ns](std::vector<double> &v, const std::vector<char> &ok)
    {
        arma::uword k0 = 0;
        while (k0 < ns && !ok[k0])
            ++k0;
        if (k0 == ns)
            return;
        for (arma::uword k = 0; k < k0; ++k)
            v[k] = v[k0];
        for (arma::uword k = k0 + 1; k < ns; ++k)
            if (!ok[k])
                v[k] = v[k - 1];
    };
    fill(heading, horizontal);
    fill(tilt, moves);
    return std::find(moves.begin(), moves.end(), 1) != moves.end();
}

// Orientation (bank 0, tilt, heading) of the n rows of a track from its steps; the last step is repeated, for a closed track
// the first row is repeated. Returns false (and leaves 'ori' alone) if the track does not move
bool track_orientation(const arma::mat &pos, arma::uword first, arma::uword n, arma::mat &ori)
{
    std::vector<double> h, t;
    if (!step_directions(pos, first, n, h, t))
        return false;
    ori.set_size(3, n);
    for (arma::uword k = 0; k < n; ++k)
    {
        const arma::uword s = std::min(k, n - 2);
        ori(0, k) = 0.0, ori(1, k) = t[s], ori(2, k) = h[s];
    }
    if (n > 2 && pos(0, first) == pos(0, first + n - 1) && pos(1, first) == pos(1, first + n - 1) && pos(2, first) == pos(2, first + n - 1))
        ori.col(n - 1) = ori.col(0); // Closed (the track moves)
    return true;
}

// Unit vector d of the direction of a point drop: from 'direction' if its azimuth is given (returns true), else the forward
// axis of the orientation
bool drop_direction(const quadriga_lib::layout_nodes &nodes, arma::uword row, double *d)
{
    const double az = nodes.direction.is_empty() ? NAN : nodes.direction(0, row);
    if (!std::isnan(az))
    {
        double el = nodes.direction(1, row);
        el = std::isnan(el) ? 0.0 : el;
        d[0] = std::cos(el) * std::cos(az), d[1] = std::cos(el) * std::sin(az), d[2] = std::sin(el);
        return true;
    }
    const double tilt = nodes.orientation(1, row), head = nodes.orientation(2, row);
    d[0] = std::cos(tilt) * std::cos(head), d[1] = std::cos(tilt) * std::sin(head), d[2] = -std::sin(tilt);
    return false;
}
} // namespace

/*!MD
# sample_track
Positions and orientations of a node at snapshot times

- Returns the position and the orientation of one node at each snapshot time, for point drops and tracks;
  [[get_channels_from_rays]] samples both ends of each link this way
- Without `times`, each row of the node is one snapshot
- The orientation includes the rotation of mounted antennas (`relative = 1`), so it is the orientation in the GCS

## Declaration:
```
void quadriga_lib::sample_track(
    const quadriga_lib::layout_nodes &nodes,
    arma::uword i_node,
    const arma::vec &times,
    arma::mat &pos,
    arma::mat &orientation,
    arma::uvec *i_row = nullptr);
```

## Inputs:
- **`nodes`** — The Tx nodes or the Rx nodes of a layout (`layout.tx` or `layout.rx`)
- **`i_node`** — Node index, 0-based
- **`times`** — Snapshot times in [s], finite; `[n_snap]`; empty = one snapshot per row

## Outputs:
- **`pos`** — Positions in [m]; `[3, n_snap]`
- **`orientation`** — Orientations (bank, tilt, heading) in [rad] in the GCS, including the mounting rotation;
  `[3, n_snap]`
- **`i_row`** *(optional)* — Row at or before each snapshot, 0-based, index into the arrays of `nodes` (not into the
  rows of the node); `[n_snap]`

## Example:
```
quadriga_lib::layout L = quadriga_lib::layout_csv_read("tx.csv", "rx.csv");
arma::mat pos, ori;
quadriga_lib::sample_track(L.rx, 0, arma::linspace(0.0, 1.0, 11), pos, ori);
```

## Notes:
- **Point drop** (one row): $p(t) = p_0 + v t \hat{d}$ with the speed $v$ of the row and the unit vector $\hat{d}$ of
  its direction (azimuth, elevation; an elevation of NaN is 0). Without a direction (NaN azimuth),
  $\hat{d} = (\cos\beta \cos\alpha, \cos\beta \sin\alpha, -\sin\beta)$ with the tilt $\beta$ and the heading $\alpha$ of
  the orientation; the bank has no effect. The orientation is constant
- **Track** (several rows): the time of row $i$ is `time`, else it follows from the speed $v_i$ of the row, $t_0 = 0$,
  $t_{i+1} = t_i + \lvert p_{i+1} - p_i \rvert / v_i$; a step of zero length with speed 0 takes no time. The position is
  interpolated linearly in time, bank and heading along the shorter way and the tilt linearly; before the first and
  after the last time, the first and the last row hold. If several rows have the same time, the later row holds at that
  time. The times in `times` need not increase. A track with neither `time` nor a speed throws if `times` is given; a
  moving step with speed 0 throws
- **Mounted antennas** (`relative = 1`): the orientation is relative to the direction of motion, and the output is
  $R_\mathrm{motion} R_\mathrm{orientation}$, with bank 0 in $R_\mathrm{motion}$. For a point drop, the motion is its
  direction, which must be given. For a track, it is the direction of the step from the row at or before the snapshot to
  the next row (the last step repeated at the end); a step of zero length takes the previous direction, at the start the
  next one that exists. The stored orientation is interpolated first, then rotated by the motion
- **Checks**: members of `nodes` with sizes that do not fit, and a `direction` on a track, throw
  `std::invalid_argument`; the other rules of [[layout]] are not checked

## See also:
- [[generate_track]] (creates tracks)
- [[set_orientation]] (orientation from the motion)
- [[layout]] (the members of a node)
MD!*/

void quadriga_lib::sample_track(const layout_nodes &nodes, arma::uword i_node, const arma::vec &times, arma::mat &pos,
                                arma::mat &orientation, arma::uvec *i_row)
{
    if (i_node >= nodes.name.size())
        throw std::invalid_argument("Input 'i_node' is out of range.");
    check_structure(nodes, false, i_node);
    for (arma::uword k = 0; k < times.n_elem; ++k)
        if (!std::isfinite(times[k]))
            throw std::invalid_argument("Input 'times' must be finite.");

    const arma::uword first = nodes.offset[i_node], n = nodes.n_row(i_node);
    const bool rel = is_relative(nodes, i_node);
    const arma::uword n_snap = times.is_empty() ? n : times.n_elem;
    arma::mat P(3, n_snap), O(3, n_snap);
    arma::uvec I(n_snap);

    if (n == 1) // Point drop
    {
        double d[3];
        const bool from_dir = drop_direction(nodes, first, d);
        if (rel && !from_dir)
            throw std::invalid_argument(who(nodes, i_node) + "'relative' = 1 and one row need a 'direction'.");
        double o[3] = {nodes.orientation(0, first), nodes.orientation(1, first), nodes.orientation(2, first)};
        if (rel)
        {
            double h, t;
            heading_tilt(d[0], d[1], d[2], h, t);
            mounted(h, t, o, o);
        }
        const double v = nodes.speed.is_empty() ? 0.0 : nodes.speed[first];
        for (arma::uword k = 0; k < n_snap; ++k)
        {
            const double t = times.is_empty() ? 0.0 : times[k];
            for (arma::uword a = 0; a < 3; ++a)
                P(a, k) = nodes.pos(a, first) + v * d[a] * t, O(a, k) = o[a];
            I[k] = first;
        }
    }
    else // Track
    {
        for (arma::uword r = first; r < first + n && !nodes.direction.is_empty(); ++r)
            if (!std::isnan(nodes.direction(0, r)))
                throw std::invalid_argument(who(nodes, i_node) + "a track must not have a 'direction'.");

        std::vector<double> hd, tl; // Motion of each step
        if (rel)
            step_directions(nodes.pos, first, n, hd, tl);
        auto set_orientation_of = [&](const double *o, arma::uword step, arma::uword k)
        {
            double out[3] = {o[0], o[1], o[2]};
            if (rel)
                mounted(hd[step], tl[step], o, out);
            for (arma::uword a = 0; a < 3; ++a)
                O(a, k) = out[a];
        };

        if (times.is_empty()) // One snapshot per row
        {
            for (arma::uword k = 0; k < n; ++k)
            {
                const double o[3] = {nodes.orientation(0, first + k), nodes.orientation(1, first + k), nodes.orientation(2, first + k)};
                for (arma::uword a = 0; a < 3; ++a)
                    P(a, k) = nodes.pos(a, first + k);
                set_orientation_of(o, std::min(k, n - 2), k);
                I[k] = first + k;
            }
        }
        else
        {
            // Times of the rows
            std::vector<double> tr(n);
            const bool has_time = !nodes.time.is_empty() && !std::isnan(nodes.time[first]);
            if (has_time)
            {
                for (arma::uword k = 0; k < n; ++k)
                {
                    tr[k] = nodes.time[first + k];
                    if (std::isnan(tr[k]) || (k > 0 && !(tr[k] > tr[k - 1])))
                        throw std::invalid_argument(who(nodes, i_node) + "'time' must be given on all rows and increase strictly.");
                }
            }
            else
            {
                bool has_speed = false;
                for (arma::uword k = 0; k < n && !nodes.speed.is_empty(); ++k)
                    has_speed = has_speed || nodes.speed[first + k] != 0.0;
                if (!has_speed)
                    throw std::invalid_argument(who(nodes, i_node) + "a track needs 'time' or 'speed' to be sampled at times.");
                tr[0] = 0.0;
                for (arma::uword k = 0; k + 1 < n; ++k)
                {
                    const double len = arma::norm(nodes.pos.col(first + k + 1) - nodes.pos.col(first + k));
                    const double v = nodes.speed[first + k];
                    if (len > 0.0 && !(v > 0.0))
                        throw std::invalid_argument(who(nodes, i_node) + "a step moves, but its 'speed' is 0.");
                    tr[k + 1] = tr[k] + (len > 0.0 ? len / v : 0.0);
                }
            }

            for (arma::uword q = 0; q < n_snap; ++q)
            {
                const double t = times[q];
                const arma::uword up = (arma::uword)(std::upper_bound(tr.begin(), tr.end(), t) - tr.begin()); // Rows at or before t
                if (up == 0 || up == n) // Before the first or after the last time: hold
                {
                    const arma::uword k = (up == 0) ? 0 : n - 1;
                    const double o[3] = {nodes.orientation(0, first + k), nodes.orientation(1, first + k), nodes.orientation(2, first + k)};
                    for (arma::uword a = 0; a < 3; ++a)
                        P(a, q) = nodes.pos(a, first + k);
                    set_orientation_of(o, std::min(k, n - 2), q);
                    I[q] = first + k;
                }
                else
                {
                    const arma::uword k = up - 1;
                    const double w = (t - tr[k]) / (tr[k + 1] - tr[k]);
                    double o[3];
                    for (arma::uword a = 0; a < 3; ++a)
                    {
                        P(a, q) = nodes.pos(a, first + k) + w * (nodes.pos(a, first + k + 1) - nodes.pos(a, first + k));
                        o[a] = lerp_angle(nodes.orientation(a, first + k), nodes.orientation(a, first + k + 1), w, a != 1);
                    }
                    set_orientation_of(o, k, q);
                    I[q] = first + k;
                }
            }
        }
    }
    pos = std::move(P);
    orientation = std::move(O);
    if (i_row != nullptr)
        *i_row = std::move(I);
}

/*!MD
# set_orientation
Set the orientation of nodes from their motion, or to a given value

- Turns the antennas of tracks along their motion, e.g. for a UT held in the direction of travel, or sets all rows of
  the nodes to a fixed orientation
- `force = false`: tracks get bank 0 and the tilt and heading of the steps between their rows, plus `orientation` as an
  offset; point drops keep their orientation plus the offset
- `force = true`: every row of the nodes gets exactly `orientation`

## Declaration:
```
void quadriga_lib::set_orientation(
    quadriga_lib::layout_nodes &nodes,
    const arma::vec &orientation = {},
    const uint_list &i_node = {},
    bool force = false);
```

## Inputs:
- **`nodes`** — The Tx nodes or the Rx nodes of a layout (`layout.tx` or `layout.rx`), updated in place
- **`orientation`** *(optional)* — Bank, tilt, heading in [rad], 3 finite values; empty = 0
- **`i_node`** *(optional)* — Node indices, 0-based; empty = all nodes
- **`force`** *(optional)* — `false`: orientation from the motion of tracks, plus `orientation` as offset; `true`:
  `orientation` on all rows

## Example:
```
quadriga_lib::layout L = quadriga_lib::layout_csv_read("tx.csv", "rx.csv");
quadriga_lib::set_orientation(L.rx);                                  // The antennas look along the motion
quadriga_lib::set_orientation(L.rx, arma::vec{0.0, 0.0, 0.5});        // The same, turned by 0.5 rad
```

## Notes:
- **Tracks** (`force = false`): forward differences between the rows, the last step repeated, or for a closed track the
  step to the first row, as `qd_track.calc_orientation` of the NG library does; the angles are wrapped to [-π, π]. A
  step of zero length keeps the previous orientation, at the start the next one that exists. A vertical step has the
  tilt -90° (up) or 90° (down) and keeps the previous heading (a deliberate deviation from NG, where it is 0). A rising
  track has a negative tilt; a track that does not move keeps its orientation (plus the offset)
- **Point drops** (`force = false`): without an offset, the orientation is not touched; random orientations are not
  generated here
- **Mounted antennas**: `force = false` leaves nodes with `relative = 1` unchanged, offset included, since their
  orientation is relative to the motion, which [[sample_track]] applies at each snapshot; `force = true` sets them too
- A failed call leaves `nodes` unchanged

## See also:
- [[generate_track]] (creates tracks with this orientation)
- [[sample_track]] (applies the motion to mounted antennas)
MD!*/

void quadriga_lib::set_orientation(layout_nodes &nodes, const arma::vec &orientation, const uint_list &i_node, bool force)
{
    check_structure(nodes);
    if (orientation.n_elem != 0 && orientation.n_elem != 3)
        throw std::invalid_argument("Input 'orientation' must be empty or have 3 elements (bank, tilt, heading).");
    if (!orientation.is_finite())
        throw std::invalid_argument("Input 'orientation' must be finite.");
    const double off[3] = {orientation.n_elem ? orientation[0] : 0.0, orientation.n_elem ? orientation[1] : 0.0,
                           orientation.n_elem ? orientation[2] : 0.0};
    const bool zero_offset = off[0] == 0.0 && off[1] == 0.0 && off[2] == 0.0;

    const arma::uword nn = nodes.n_node();
    std::vector<arma::uword> sel;
    if (i_node.vec().is_empty())
        for (arma::uword i = 0; i < nn; ++i)
            sel.push_back(i);
    else
        for (arma::uword i : i_node.vec())
        {
            if (i >= nn)
                throw std::invalid_argument("Input 'i_node' is out of range.");
            sel.push_back(i);
        }

    arma::mat O = nodes.orientation; // Written back at the end
    for (arma::uword i : sel)
    {
        const arma::uword first = nodes.offset[i], n = nodes.n_row(i);
        if (force)
        {
            for (arma::uword r = first; r < first + n; ++r)
                for (arma::uword a = 0; a < 3; ++a)
                    O(a, r) = off[a];
        }
        else if (is_relative(nodes, i))
            continue;
        else if (n == 1)
        {
            if (!zero_offset)
                for (arma::uword a = 0; a < 3; ++a)
                    O(a, first) = wrap_pi(O(a, first) + off[a]);
        }
        else
        {
            arma::mat T;
            if (!track_orientation(nodes.pos, first, n, T)) // A track that does not move keeps its orientation
                T = nodes.orientation.cols(first, first + n - 1);
            for (arma::uword k = 0; k < n; ++k)
                for (arma::uword a = 0; a < 3; ++a)
                    O(a, first + k) = zero_offset ? T(a, k) : wrap_pi(T(a, k) + off[a]);
        }
    }
    nodes.orientation = std::move(O);
}

namespace
{
// Points at the arc lengths s_k = k L / n_seg, k = 0 ... n_seg, of the polyline 'pl' (complex plane, at least L long), [3, n_seg + 1]
arma::mat resample_polyline(const std::vector<std::complex<double>> &pl, double length, arma::uword n_seg)
{
    arma::mat out(3, n_seg + 1, arma::fill::zeros);
    arma::uword seg = 0;
    double s0 = 0.0, s1 = std::abs(pl[1] - pl[0]); // Arc length at the start and at the end of the segment 'seg'
    for (arma::uword k = 0; k <= n_seg; ++k)
    {
        const double s = length * (double)k / (double)n_seg;
        while (s > s1 && seg + 2 < pl.size())
        {
            ++seg;
            s0 = s1;
            s1 += std::abs(pl[seg + 1] - pl[seg]);
        }
        const double w = (s1 > s0) ? std::min(1.0, (s - s0) / (s1 - s0)) : 0.0;
        const std::complex<double> p = pl[seg] + w * (pl[seg + 1] - pl[seg]);
        out(0, k) = p.real(), out(1, k) = p.imag();
    }
    return out;
}

// Number of steps of a track of the given length: ceil(length / spacing), at least n_min
arma::uword n_steps(double length, double spacing, arma::uword n_min)
{
    const double n = std::ceil(length / spacing);
    if (!(n <= 1.0e7))
        throw std::invalid_argument("Inputs 'length' and 'spacing' give too many points (more than 1e7).");
    return std::max(n_min, (arma::uword)n);
}

// Points of a street relative to the start, [3, n]; the route as in qd_track.generate of the NG library
arma::mat street_points(std::mt19937_64 &rng, double length, double spacing, double direction, double len_min, double len_mu,
                        double len_std, double radius, double p_turn)
{
    using cplx = std::complex<double>;
    const cplx j(0.0, 1.0);
    const arma::uword n_seg = n_steps(length, spacing, 1); // Before the route is drawn: it throws for too many points
    double diro = 0.0, acc = 0.0;
    std::vector<cplx> pl = {cplx(0.0, 0.0)};
    while (acc < 1.1 * length)
    {
        if (pl.size() > 20000000)
            throw std::invalid_argument("The street options give too many streets (more than 2e7) for the length.");
        double street = qd_rng_normal(rng) * len_std + len_mu;
        for (int guard = 0; street < len_min; ++guard)
        {
            if (guard > 1000000)
                throw std::invalid_argument("No street length of at least 'street_length_min' could be drawn; lower the option.");
            street = qd_rng_normal(rng) * len_std + len_mu;
        }
        acc += street;
        const cplx u = std::exp(j * diro);
        pl.push_back(pl.back() + u * street);
        if (qd_rng_uniform(rng) < p_turn) // A turn at the crossing; the new direction stays within 105 deg of the start direction
        {
            double dirn = 0.0, dirb = pi;
            for (int guard = 0; dirb >= pi / 2.0 + pi / 12.0; ++guard)
            {
                if (guard > 1000000)
                    throw std::invalid_argument("No direction for the turn of a street could be drawn; check the street options.");
                const double side = qd_rng_uniform(rng) - 0.5 < 0.0 ? -1.0 : 1.0; // Separate statements: the order of the draws is fixed
                dirn = diro + side * pi / 2.0 + qd_rng_normal(rng) * pi / 12.0;
                dirb = std::abs(wrap_pi(dirn));
            }
            pl.push_back(pl.back() + radius * (u + std::exp(j * dirn)));
            diro = dirn;
        }
    }
    const cplx rot = std::exp(j * direction);
    for (cplx &p : pl)
        p *= rot;
    return resample_polyline(pl, length, n_seg);
}

// Value of an option of the street type: one finite number in [0, hi]
double option_value(const std::string &key, const std::vector<double> &v, double hi)
{
    if (v.size() != 1 || !std::isfinite(v[0]) || !(v[0] >= 0.0 && v[0] <= hi))
        throw std::invalid_argument("Option '" + key + "' must be one finite number" + (hi == 1.0 ? " in [0, 1]." : " >= 0."));
    return v[0];
}
} // namespace

/*!MD
# generate_track
Replace the rows of nodes by a generated track or a point drop, or append a node with one

- Gives nodes a straight, circular or street-like track, or makes them point drops with a speed and a direction, the
  short-term mobility of 3GPP TR 38.901
- All nodes with the name of a given node (the sectors of a site, the panels of a terminal) get the same track. It
  starts at the first position of the node and is horizontal at the height of the first row
- The antennas look along the motion, as [[set_orientation]] turns them; [[sample_track]] gives the positions at
  snapshot times
- An empty `i_node` appends a new node at `start` and gives it the track (see Notes)

## Declaration:
```
void quadriga_lib::generate_track(
    quadriga_lib::layout_nodes &nodes,
    const uint_list &i_node,
    const std::string &type,
    double length = 0.0,
    double speed = NAN,
    const arma::vec &direction = {},
    double spacing = NAN,
    const std::unordered_map<std::string, std::vector<double>> &options = {},
    arma::sword seed = -1,
    const std::string &name = "",
    const arma::vec &start = {});
```

## Inputs:
- **`nodes`** — The Tx nodes or the Rx nodes of a layout (`layout.tx` or `layout.rx`), updated in place
- **`i_node`** — Node indices, 0-based; empty: append a new node
- **`type`** — Type of the track, not case sensitive:
  - `point`: a point drop with `speed` and a horizontal `direction`; the rows after the first are removed; `length` is
    not used
  - `linear`: a straight track of `length` (default 1 m) in the start direction
  - `circular`: a closed circle of circumference `length` (default 20π m, radius 10 m); the first and the last point are
    equal
  - `street`: the route of a car through a city grid of straight streets and crossings, of `length` (default 1000 m),
    see Notes
- **`length`** *(optional)* — Length in [m], finite and >= 0; 0 = default of the type
- **`speed`** *(optional)* — Speed in [m/s], finite and >= 0, and > 0 for tracks; NaN = the speed of the first row of
  the node if it is valid (> 0 for a track); without one, `point` gets 0 and the other types throw
- **`direction`** *(optional)* — Start direction (azimuth) in [rad], finite; one value for all nodes, or one per entry
  of `i_node` (nodes of one name need equal values); empty = uniform random in [0, 2π)
- **`spacing`** *(optional)* — Point spacing in [m], > 0; NaN = two points (`linear`), 129 points (`circular`), 1 m
  (`street`)
- **`options`** *(optional)* — Options of `street`, one value each: `street_length_min` = 50, `street_length_mu` = 187,
  `street_length_std` = 83 (all in [m], >= 0; with a std of 0, mu must be >= min and > 0), `curve_radius` = 10 [m],
  `turn_probability` = 0.5 (in [0, 1]); other keys, and these keys for the other types, throw
- **`seed`** *(optional)* — Seed, >= 0 for reproducible draws; -1 = system random device; below -1 throws
- **`name`** *(optional)* — Name of the new node (empty `i_node` only); empty: the next free name
- **`start`** *(optional)* — First position `[x, y, z]` of the new node in [m] (empty `i_node` only); empty: the origin

## Example:
```
quadriga_lib::layout L = quadriga_lib::layout_csv_read("", "rx.csv");
quadriga_lib::generate_track(L.rx, {0}, "street", 500.0, 10.0, {}, NAN, {{"turn_probability", {0.3}}}, 42);
quadriga_lib::generate_track(L.rx, {}, "linear", 100.0, 1.5, {0.0}, NAN, {}, 42, "walker", {10.0, 20.0, 1.5});
```

## Notes:
- **Points**: `linear` has two points, or `ceil(length / spacing) + 1` equidistant points; `circular` 129 points, or
  `max(3, ceil(length / spacing)) + 1`; `street` is sampled with the point spacing `spacing`, and the chords between the
  points are slightly shorter than the route at the corners
- **Direction of a circle**: the angle of the start point seen from the centre of the circle. A positive value runs
  counter-clockwise, a negative one clockwise: 0 starts in the east of the circle and runs north, -π/2 starts in the
  south and runs west. A random direction is drawn in [0, 2π), so random circles run counter-clockwise
- **Street route** (as `qd_track.generate` of the NG library): straight streets of random length (normal distribution,
  drawn again below `street_length_min`) are separated by crossings. At a crossing the car turns with the probability
  `turn_probability` by about 90° (standard deviation 15°) to the left or right, cutting the corner along a path of
  length $2 r \cos(\varphi / 2)$ for the turn angle $\varphi$ and the `curve_radius` $r$. The direction stays within
  105° of the start direction, so that the car does not drive in circles
- **Values of the new rows**: `linear`, `circular` and `street` give tracks of equal `speed` without `direction`,
  oriented along the motion as [[set_orientation]] with `force = false`; nodes with `relative = 1` keep the orientation
  of their first row on all rows. `point` keeps the orientation of the node. `scenario`, `indoor`, `floor`, `d2d_in` and
  the extra columns take the values of the first row; `time` is cleared
- **Reproducibility**: a seed >= 0 gives the same track on all platforms. Each name (all its nodes) has its own random
  stream, so the track does not depend on the other nodes or the order in `i_node`
- **Append mode**: with an empty `i_node`, a new node is appended with one row at `start` and the default values (sector
  0, the generic antenna, orientation 0, speed 0, no scenario), and gets the track; a track type therefore needs
  `speed`. Its index is `nodes.n_node() - 1` after the call. The default name is the common prefix of the names before
  their number with the next number (`UT3` after `UT1`, `UT2`), else `N<n>`. `name` or `start` with a non-empty `i_node`
  throws
- A failed call leaves `nodes` unchanged

## See also:
- [[sample_track]] (snapshots of a track)
- [[set_orientation]] (orientation from the motion)
- [[layout]] (the members of a node)
MD!*/

void quadriga_lib::generate_track(layout_nodes &nodes, const uint_list &i_node, const std::string &type_in, double length, double speed,
                                  const arma::vec &direction, double spacing,
                                  const std::unordered_map<std::string, std::vector<double>> &options, arma::sword seed,
                                  const std::string &name, const arma::vec &start)
{
    check_structure(nodes);
    if (!i_node.vec().is_empty() && (!name.empty() || !start.is_empty()))
        throw std::invalid_argument("Inputs 'name' and 'start' are used only when 'i_node' is empty.");
    if (i_node.vec().is_empty()) // Append mode: a new node, then its track; a failed call leaves 'nodes' unchanged
    {
        layout_nodes N = nodes;
        append_node(N, name, start);
        generate_track(N, N.n_node() - 1, type_in, length, speed, direction, spacing, options, seed);
        nodes = std::move(N);
        return;
    }

    std::string type = type_in;
    std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    if (type != "point" && type != "linear" && type != "circular" && type != "street")
        throw std::invalid_argument("Input 'type' must be 'point', 'linear', 'circular' or 'street' ('" + type_in + "').");
    const bool is_point = type == "point", is_street = type == "street";

    const arma::uvec &sel = i_node.vec();
    if (sel.is_empty())
        throw std::invalid_argument("Input 'i_node' must not be empty.");
    const arma::uword nn = nodes.n_node();
    for (arma::uword i : sel)
        if (i >= nn)
            throw std::invalid_argument("Input 'i_node' is out of range.");

    if (!is_point && !(std::isfinite(length) && length >= 0.0))
        throw std::invalid_argument("Input 'length' must be finite and >= 0.");
    if (!std::isnan(speed) && !(std::isfinite(speed) && speed >= 0.0))
        throw std::invalid_argument("Input 'speed' must be NaN, or finite and >= 0.");
    if (!std::isnan(speed) && !is_point && speed == 0.0)
        throw std::invalid_argument("Input 'speed' must be > 0 for a track, as its steps move.");
    if (seed < -1)
        throw std::invalid_argument("Input 'seed' must be >= -1.");
    if (!std::isnan(spacing) && !(std::isfinite(spacing) && spacing > 0.0))
        throw std::invalid_argument("Input 'spacing' must be NaN, or finite and > 0.");
    if (!direction.is_empty() && direction.n_elem != 1 && direction.n_elem != sel.n_elem)
        throw std::invalid_argument("Input 'direction' must be empty, have one element, or one per entry of 'i_node'.");
    if (!direction.is_finite())
        throw std::invalid_argument("Input 'direction' must be finite.");

    // Options of the type
    double len_min = 50.0, len_mu = 187.0, len_std = 83.0, radius = 10.0, p_turn = 0.5;
    const double inf = arma::datum::inf;
    for (const auto &kv : options)
    {
        const std::string &k = kv.first;
        if (k != "street_length_min" && k != "street_length_mu" && k != "street_length_std" && k != "curve_radius" && k != "turn_probability")
            throw std::invalid_argument("Input 'options' has an unknown key ('" + k + "').");
        if (!is_street)
            throw std::invalid_argument("Input 'options' has the key '" + k + "', which the type '" + type + "' does not use.");
        if (k == "street_length_min")
            len_min = option_value(k, kv.second, inf);
        else if (k == "street_length_mu")
            len_mu = option_value(k, kv.second, inf);
        else if (k == "street_length_std")
            len_std = option_value(k, kv.second, inf);
        else if (k == "curve_radius")
            radius = option_value(k, kv.second, inf);
        else
            p_turn = option_value(k, kv.second, 1.0);
    }
    if (is_street && len_std == 0.0 && !(len_mu >= len_min && len_mu > 0.0))
        throw std::invalid_argument("Options 'street_length_mu' and 'street_length_min': with 'street_length_std' = 0, no street of positive length >= 'street_length_min' can be drawn.");

    // The start direction of each name, in the order of first appearance in i_node
    std::vector<std::string> names;
    std::unordered_map<std::string, double> given;
    for (arma::uword k = 0; k < sel.n_elem; ++k)
    {
        const std::string &nm = nodes.name[sel[k]];
        const double dk = direction.is_empty() ? NAN : direction[direction.n_elem == 1 ? 0 : k];
        if (!given.count(nm))
        {
            names.push_back(nm);
            given[nm] = dk;
        }
        else if (!(dk == given[nm]) && !(std::isnan(dk) && std::isnan(given[nm])))
            throw std::invalid_argument("Input 'direction' has different values for nodes of the name '" + nm + "'.");
    }

    // The points of the tracks relative to the first position, before anything is changed
    struct gen
    {
        double az = 0.0;
        arma::mat rel; // [3, n]
    };
    std::unordered_map<std::string, gen> G;
    for (const std::string &nm : names)
    {
        // One stream per name: std::mt19937_64 seeded with the hash of the seed and the key. Uniform numbers use the upper 53
        // bits of an output, normal numbers the Box-Muller method (qd_rng_*), so a seed gives the same track on all platforms.
        // The first draw is the start direction; it is drawn also when the direction is given, so that the shape of a street
        // for a seed does not depend on it
        std::mt19937_64 rng = qd_rng(seed >= 0 ? qd_layout_seed_hash(seed, "track|" + nm) : (arma::sword)-1);
        gen g;
        const double u = qd_rng_uniform(rng);
        g.az = std::isnan(given[nm]) ? two_pi * u : given[nm]; // Random: [0, 2 pi), so random circles run counter-clockwise
        if (is_point)
        {
            g.az = wrap_pi(g.az);
            g.rel.zeros(3, 1);
        }
        else if (type == "linear")
        {
            const double L = length > 0.0 ? length : 1.0;
            const arma::uword ns = std::isnan(spacing) ? 1 : n_steps(L, spacing, 1);
            g.rel.zeros(3, ns + 1);
            for (arma::uword k = 0; k <= ns; ++k)
            {
                g.rel(0, k) = L * (double)k / (double)ns * std::cos(g.az);
                g.rel(1, k) = L * (double)k / (double)ns * std::sin(g.az);
            }
        }
        else if (type == "circular")
        {
            const double L = length > 0.0 ? length : two_pi * 10.0, r = L / two_pi;
            const arma::uword ns = std::isnan(spacing) ? 128 : n_steps(L, spacing, 3);
            const double sgn = g.az >= 0.0 ? 1.0 : -1.0; // The sign of the direction as given sets the sense of rotation
            g.rel.zeros(3, ns + 1); // The last point stays 0: the circle is closed
            for (arma::uword k = 0; k < ns; ++k)
            {
                const double phi = g.az + sgn * two_pi * (double)k / (double)ns;
                g.rel(0, k) = r * (std::cos(phi) - std::cos(g.az));
                g.rel(1, k) = r * (std::sin(phi) - std::sin(g.az));
            }
        }
        else
            g.rel = street_points(rng, length > 0.0 ? length : 1000.0, std::isnan(spacing) ? 1.0 : spacing, g.az, len_min, len_mu,
                                  len_std, radius, p_turn);
        G[nm] = std::move(g);
    }

    // Speed of each changed node
    std::vector<char> is_gen(nn, 0);
    std::vector<double> node_speed(nn, 0.0);
    bool any_speed = false;
    for (arma::uword i = 0; i < nn; ++i)
    {
        is_gen[i] = G.count(nodes.name[i]) ? 1 : 0;
        if (!is_gen[i])
            continue;
        const double old = nodes.speed.is_empty() ? 0.0 : nodes.speed[nodes.offset[i]];
        const bool old_ok = std::isfinite(old) && old > 0.0;
        if (!std::isnan(speed))
            node_speed[i] = speed;
        else if (old_ok)
            node_speed[i] = old;
        else if (!is_point)
            throw std::invalid_argument(who(nodes, i) + "input 'speed' is NaN, but the first row has no speed > 0 that a track can keep.");
        any_speed = any_speed || node_speed[i] != 0.0;
    }

    // The new rows: the row of the old arrays that gives the other members
    std::vector<arma::uword> src, n_new(nn);
    arma::uvec offset(nn);
    for (arma::uword i = 0; i < nn; ++i)
    {
        offset[i] = src.size();
        const arma::uword first = nodes.offset[i];
        n_new[i] = is_gen[i] ? G[nodes.name[i]].rel.n_cols : nodes.n_row(i);
        for (arma::uword r = 0; r < n_new[i]; ++r)
            src.push_back(is_gen[i] ? first : first + r);
    }
    const arma::uword N = (arma::uword)src.size();

    arma::mat pos(3, N), ori(3, N), dir;
    arma::vec spd, tim, d2d;
    arma::s32_vec ind, flr;
    std::vector<std::string> scen;
    std::vector<std::vector<std::string>> extra(nodes.extra.size());
    const bool want_dir = !nodes.direction.is_empty() || is_point, want_speed = !nodes.speed.is_empty() || any_speed;
    if (want_dir)
        dir.set_size(2, N);
    if (want_speed)
        spd.set_size(N);
    if (!nodes.time.is_empty())
        tim.set_size(N);
    if (!nodes.d2d_in.is_empty())
        d2d.set_size(N);
    if (!nodes.indoor.is_empty())
        ind.set_size(N);
    if (!nodes.floor.is_empty())
        flr.set_size(N);
    if (!nodes.scenario.empty())
        scen.resize(N);
    for (auto &e : extra)
        e.resize(N);
    for (arma::uword r = 0; r < N; ++r)
    {
        const arma::uword s = src[r];
        pos.col(r) = nodes.pos.col(s);
        ori.col(r) = nodes.orientation.col(s);
        if (want_dir)
            dir(0, r) = nodes.direction.is_empty() ? NAN : nodes.direction(0, s), dir(1, r) = nodes.direction.is_empty() ? NAN : nodes.direction(1, s);
        if (want_speed)
            spd[r] = nodes.speed.is_empty() ? 0.0 : nodes.speed[s];
        if (!tim.is_empty())
            tim[r] = nodes.time[s];
        if (!d2d.is_empty())
            d2d[r] = nodes.d2d_in[s];
        if (!ind.is_empty())
            ind[r] = nodes.indoor[s];
        if (!flr.is_empty())
            flr[r] = nodes.floor[s];
        if (!scen.empty())
            scen[r] = nodes.scenario[s];
        for (size_t e = 0; e < extra.size(); ++e)
            extra[e][r] = nodes.extra[e][s];
    }

    // The generated nodes
    for (arma::uword i = 0; i < nn; ++i)
    {
        if (!is_gen[i])
            continue;
        const gen &g = G[nodes.name[i]];
        const arma::uword n = n_new[i], o = offset[i];
        const arma::vec p0 = nodes.pos.col(nodes.offset[i]);
        for (arma::uword k = 0; k < n; ++k)
        {
            pos.col(o + k) = p0 + g.rel.col(k);
            if (want_speed)
                spd[o + k] = node_speed[i];
            if (!tim.is_empty())
                tim[o + k] = NAN;
            if (want_dir)
                dir(0, o + k) = is_point ? g.az : NAN, dir(1, o + k) = is_point ? 0.0 : NAN;
        }
        if (!is_point && !is_relative(nodes, i)) // The orientation follows the motion (as set_orientation with force = false)
        {
            arma::mat T;
            if (track_orientation(pos, o, n, T))
                ori.cols(o, o + n - 1) = T;
        }
    }
    auto all_nan = [](const double *x, arma::uword n, arma::uword stride)
    {
        for (arma::uword k = 0; k < n; ++k)
            if (!std::isnan(x[k * stride]))
                return false;
        return true;
    };
    if (!tim.is_empty() && all_nan(tim.memptr(), tim.n_elem, 1))
        tim.reset(); // No node has a time any more
    if (!dir.is_empty() && all_nan(dir.memptr(), dir.n_cols, 2))
        dir.reset(); // No node has a direction any more

    nodes.pos = std::move(pos);
    nodes.orientation = std::move(ori);
    nodes.direction = std::move(dir);
    nodes.speed = std::move(spd);
    nodes.time = std::move(tim);
    nodes.d2d_in = std::move(d2d);
    nodes.indoor = std::move(ind);
    nodes.floor = std::move(flr);
    nodes.scenario = std::move(scen);
    nodes.extra = std::move(extra);
    nodes.offset = std::move(offset);
}
