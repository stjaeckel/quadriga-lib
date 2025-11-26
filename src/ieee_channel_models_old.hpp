// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#ifndef qdlib_ieee_channel_models_H
#define qdlib_ieee_channel_models_H

#include <armadillo>
#include <string>
#include <vector>
#include <cmath>
#include <complex>



void qd_ieee_chan_old(std::string ChannelType,        // Channel Model Type (A, B, C, D, E, F) as defined by TGn
                      arma::uword n_tx = 1,           // Number of transmit antennas
                      arma::uword n_rx = 1,           // Number of receive antennas
                      arma::uword n_users = 1,        // Number of user (only for TGac)
                      bool downlink = true,           // Switch for uplink / downlink direction
                      arma::vec Dist_m = {4.99},      // Distance between TX and TX in meters, length n_users or length 1 (if same for all users)
                      double CarrierFreq_Hz = 5.25e9, // Carrier frequency in Hz
                      double tap_spacing_ns = 10.0,   // Taps spacing in ns, must be equal to 10 divided by a power of 2
                      double PowerLineFreq = 50.0,    // Power line frequency in Hz for Doppler Components Due to Fluorescent Lights in models D and E
                      double Tx_spacing = 0.5,        // Transmit antenna spacing in [lambda]
                      double Rx_spacing = 0.5,        // Receive antenna spacing in [lambda]
                      double AoD_offset = 0.0,        // AoD offset in deg, only used for n_users == 1
                      double AoA_offset = 0.0,        // AoA offset in deg, only used for nUsers == 1
                      int sdma_model = 1              // Offset model, 0 = None (LOS steering only), 1 = All clusters, 2 = First cluster only
)
{
    if (n_tx == 0 || n_rx == 0 || n_users == 0)
        throw std::invalid_argument("Number of Tx/Rx/Users antennas cannot be 0.");

    if (Dist_m.n_elem != 1 && Dist_m.n_elem != n_users)
        throw std::invalid_argument("Dist_m must be a vector of length 1 or length n_users.");
    if (arma::any(Dist_m <= 0.0))
        throw std::invalid_argument("TX-RX distance cannot be 0 or negative.");
    if (Dist_m.n_elem == 1 && n_users != 1)
        Dist_m = arma::vec(n_users, arma::fill::value(Dist_m[0]));

    if (CarrierFreq_Hz <= 0.1)
        throw std::invalid_argument("Invalid carrier frequency.");

    double tmp;
    double bw_factor = 10.0 / tap_spacing_ns; // bandwidth expansion factor
    if (std::abs(std::modf(std::log2(bw_factor), &tmp)) > 1e-9 || tap_spacing_ns > 10.0 || tap_spacing_ns <= 0.0)
        throw std::invalid_argument("Tap spacing must be equal to 10 divided by a power of 2.");

    if (PowerLineFreq <= 0.1)
        throw std::invalid_argument("Invalid power line frequency.");

    if (Tx_spacing < 0.0)
        throw std::invalid_argument("Invalid transmit antenna spacing.");

    if (Rx_spacing < 0.0)
        throw std::invalid_argument("Invalid receive antenna spacing.");

    // ---------- SCENARIO PARAMETERS ----------

    // Model parameters as in IEEE 802.11-03/940r4, Appendix C
    arma::mat power_clst_dB;     // Cluster power, rows = clusters, cols = paths
    arma::mat AoD_deg;           // Departure angles at the RX
    arma::mat ASD_deg;           // Angular spread at the TX
    arma::mat AoA_deg;           // Arrival angles at the RX
    arma::mat ASA_deg;           // Angular spread at the RX
    arma::vec power_avg_dB;      // Average path power in dB
    arma::vec delay_ns;          // Path delay in ns
    double dBP_m = 0.0;          // Path loss break point in meters
    double SF_std_dB_NLOS = 3.0; // Shadow Fading STD for NLOS channels in dB
    double KF_linear = 1.0;      // Boost of LOS power at small distances below break point

    if (ChannelType == "B")
    {
        power_clst_dB = {{0.0, -5.4287, -10.8574, -16.2860, -21.7147, -INFINITY, -INFINITY, -INFINITY, -INFINITY},
                         {-INFINITY, -INFINITY, -3.2042, -6.3063, -9.4084, -12.5105, -15.6126, -18.7147, -21.8168}};

        AoD_deg = {{225.1084, 225.1084, 225.1084, 225.1084, 225.1084, 0.0, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 106.5545, 106.5545, 106.5545, 106.5545, 106.5545, 106.5545, 106.5545}};

        ASD_deg = {{14.4699, 14.4699, 14.4699, 14.4699, 14.4699, 0.0, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 25.2566, 25.2566, 25.2566, 25.2566, 25.2566, 25.2566, 25.2566}};

        AoA_deg = {{4.3943, 4.3943, 4.3943, 4.3943, 4.3943, 0.0, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 118.4327, 118.4327, 118.4327, 118.4327, 118.4327, 118.4327, 118.4327}};

        ASA_deg = {{14.4699, 14.4699, 14.4699, 14.4699, 14.4699, 0.0, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 25.2566, 25.2566, 25.2566, 25.2566, 25.2566, 25.2566, 25.2566}};

        power_avg_dB = {0.0, -5.4287, -2.5162, -5.8905, -9.1603, -12.5105, -15.6126, -18.7147, -21.8168};
        delay_ns = {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0};

        dBP_m = 5.0;
        SF_std_dB_NLOS = 4.0;
        KF_linear = 1.0;
    }
    else
        throw std::invalid_argument("Invalid ChannelType: " + ChannelType);

    // ---------- HELPER FUNCTIONS ----------

    // Constants
    const double pi = 3.141592653589793;
    const double sqrt2 = std::sqrt(2.0);
    const double isqrt2 = 1.0 / sqrt2;
    const double sqrt8 = std::sqrt(8.0);

    auto dB_2_linear = [](double x)
    { return std::pow(10.0, x * 0.1); };

    auto rad = [](double deg)
    { return deg * 0.017453292519943; };

    auto deg = [](double rad)
    { return rad * 57.295779513082323; };

    auto find_index = [](size_t n_data, const double *data, double val) -> size_t
    {
        for (size_t i_data = 0; i_data < n_data; ++i_data)
            if (data[i_data] >= val)
                return i_data;
        return n_data;
    };

    auto hermitian_toeplitz_matrix = [](size_t n_val,              // Number of values in re and im
                                        const double *re,          // Real part of the input vector, length n_val
                                        const double *im,          // Imaginary part of the input vector, length n_val
                                        std::complex<double> *res) // Output: Hermitian Toeplitz, col-major, length n_val^2
    {
        for (size_t col = 0; col < n_val; ++col)
            for (size_t row = 0; row < n_val; ++row)
            {
                const size_t k = (col >= row) ? (col - row) : (row - col);
                if (row <= col)
                    res[col * n_val + row] = {re[k], im[k]}; // upper (incl. diag): given row
                else
                    res[col * n_val + row] = {re[k], -im[k]}; // lower: conjugate
            }
    };

    // ---------- MODEL FUNCTIONS ----------

    // Lambda function for offset angle generation
    // see IEEE 802.11-09/0308r12, A.2 – MATLAB-Independent Implementation
    auto get_offset_angles = [](arma::uword seed, arma::uword nValues, double *ptr)
    {
        arma::uword a = 16807;      // (7^5)
        arma::uword m = 2147483647; // (2^31-1)
        double md = 1.0 / (double)m;

        for (arma::uword i = 0; i < nValues; ++i)
        {
            double rand_variant = (double)seed * md;
            ptr[i] = (rand_variant - 0.5) * 360.0;
            seed = (a * seed) % m;
        }
    };

    // Lookup-table for AS to sigma_laplacian mapping
    // See: IEEE 802.11-03/940r4, Sec. 4.2 PAS Shape
    constexpr double pas_map_AS[129] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 6.99999999999961, 7.99999999996736,
                                        8.99999999899714, 9.99999998461741, 10.9999998574985, 11.9999990949232, 12.9999956976519, 13.9999837054695, 14.9999485261405,
                                        15.9998596411514, 16.9996608592331, 17.9992589520795, 18.9985119823364, 19.9972187875949, 20.995110918066, 21.9918479364342,
                                        22.9870165049242, 23.9801332199394, 24.9706507829649, 25.9579668505112, 26.9414347858725, 27.9203755214353, 28.8940898034351,
                                        29.8618702019563, 30.8230124021961, 31.7768254292421, 32.7226405848202, 33.6598189831417, 34.5877576608532, 35.5058943029479,
                                        36.4137106739593, 37.3107348745202, 38.1965425604661, 39.0707572681411, 39.9330499881729, 40.7831381230977, 41.6207839537671,
                                        42.4457927269851, 43.2580104634431, 44.0573215715887, 44.8436463401379, 45.6169383699028, 46.3771819946593, 47.1243897310273,
                                        47.8585997887944, 48.5798736657463, 49.2882938447934, 49.9839616059197, 50.6669949611134, 51.337526716866, 51.9957026659509,
                                        52.6416799079023, 53.2756252958393, 53.8977140059256, 54.5081282247543, 55.1070559492422, 55.6946898931507, 56.2712264940743,
                                        56.8368650146199, 57.3918067314977, 57.9362542063437, 58.4704106322536, 58.9944792502317, 59.5086628300083, 60.0131632099618,
                                        60.5081808911741, 60.9939146809489, 61.4705613814253, 61.9383155192148, 62.3973691122821, 62.8479114705671, 63.2901290271164,
                                        63.7242051967444, 64.1503202594861, 64.5686512663307, 64.9793719649333, 65.3826527432048, 65.7786605888584, 66.1675590631663,
                                        66.5495082873338, 66.9246649400458, 67.2931822648735, 67.6552100863522, 68.0108948336552, 68.3603795708888, 68.7038040331338,
                                        69.0413046674382, 69.3730146780515, 69.6990640752585, 70.0195797272374, 70.334685414426, 70.6445018859369, 70.9491469176061,
                                        71.2487353713093, 71.5433792552187, 71.8331877847086, 72.1182674436521, 72.3987220458834, 72.6746527966224, 72.946158353687,
                                        73.2133348883386, 73.4762761456275, 73.7350735041188, 73.9898160349028, 74.2405905597993, 74.4874817086865, 74.7305719758908,
                                        74.9699417755864, 75.2056694961646, 75.4378315535372, 75.6665024433481, 75.891754792075, 76.1136594070047, 76.3322853250757};

    constexpr double pas_map_sigma[129] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                           16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                           41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0,
                                           66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                                           91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0,
                                           114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0};

    // Map cluster powers and angle spreads to laplacian PAS and power factors
    auto laplacian_mapping = [&](size_t n_clst,         // Number of clusters
                                 const double *clst_P,  // Per-cluster-power, linear scale, length n_clst
                                 const double *clst_AS, // Per-cluster AS, degree, length n_clst
                                 double *sigma,         // Sigma_laplacian map in radians, length n_clst
                                 double *Q)             // Power normalization coefficients Q, linear, length n_clst
    {
        // Map Cluster AS to sigma_laplacian using the provided lookup tables
        for (size_t i_clst = 0; i_clst < n_clst; ++i_clst)
        {
            size_t ind_next = find_index(129, pas_map_AS, clst_AS[i_clst]);

            if (ind_next == 0) // First value
                sigma[i_clst] = pas_map_sigma[0];
            else if (ind_next == 129) // Last value
                sigma[i_clst] = pas_map_sigma[128];
            else // Linear interpolation
            {
                size_t ind_prev = ind_next - 1;
                double val1 = (pas_map_AS[ind_next] - clst_AS[i_clst]) * pas_map_sigma[ind_prev];
                double val2 = (clst_AS[i_clst] - pas_map_AS[ind_prev]) * pas_map_sigma[ind_next];
                double diff_as = pas_map_AS[ind_next] - pas_map_AS[ind_prev];
                sigma[i_clst] = (val1 + val2) / diff_as;
            }
        }

        // Convert sigma to radians
        for (size_t i_clst = 0; i_clst < n_clst; ++i_clst)
            sigma[i_clst] = rad(sigma[i_clst]);

        // Calc power normalization coefficients Q
        //      Inputs: sigma_rad, power_lin
        //      r   = (sigma_rad .* power_lin) / max(sigma_rad .* power_lin);
        //      w   = 1 - exp(-sqrt(2) * (pi ./ sigma_rad));
        //      q1  = 1 / sum(r .* w);
        //      Q   = (r * q1).';
        size_t ref_clst = 0;
        double ref = 0.0;
        for (size_t i_clst = 0; i_clst < n_clst; ++i_clst)
        {
            Q[i_clst] = sigma[i_clst] * clst_P[i_clst];
            double a = std::abs(Q[i_clst]);
            if (a > ref)
                ref = a, ref_clst = i_clst;
        }
        ref = 1.0 / Q[ref_clst];
        for (size_t i_clst = 0; i_clst < n_clst; ++i_clst)
            Q[i_clst] *= ref;

        ref = 0.0;
        for (size_t i_clst = 0; i_clst < n_clst; ++i_clst)
        {
            double tmp = 1.0 - std::exp(-4.442882938158366 / sigma[i_clst]);
            ref += Q[i_clst] * tmp;
        }
        ref = 1.0 / ref;

        for (size_t i_clst = 0; i_clst < n_clst; ++i_clst)
            Q[i_clst] *= ref;
    };

    // Compute the composite arrival / departure angle and angular spread for overlapping clusters
    auto get_composite_path_angles = [&](size_t n_clst,          // Number of clusters
                                         const double *clst_P,   // Per-cluster-power, linear scale, length n_clst
                                         const double *clst_AoX, // Per-cluster angle, degree, length n_clst
                                         const double *clst_AS,  // Per-cluster AS, degree, length n_clst
                                         double &path_AoX,       // Output: Composite path angle, degree, scalar
                                         double &path_AS)        // Output: Composite path AS, degree, scalar
    {
        // Get number of non-zero clusters
        size_t n_clst_nonzero = 0;
        size_t i_clst_nonzero = 0;
        for (size_t i_clst = 0; i_clst < n_clst; ++i_clst)
            if (clst_P[i_clst] > 1e-5)
                i_clst_nonzero = i_clst, ++n_clst_nonzero;

        // Fast exit if only 1 cluster has non-zero power
        if (n_clst_nonzero < 2)
        {
            path_AoX = clst_AoX[i_clst_nonzero];
            path_AS = clst_AS[i_clst_nonzero];
            return;
        }

        // Map cluster AS to laplacian PAS using the provided lookup tables
        std::vector<double> sigma_rad(n_clst), Q(n_clst);
        laplacian_mapping(n_clst, clst_P, clst_AS, sigma_rad.data(), Q.data());

        // Calculate composite angle
        double path_AoX_rad = 0.0;
        for (size_t i_clst = 0; i_clst < n_clst; ++i_clst)
        {
            double e_fac = 1.0 - std::exp(-4.442882938158366 / sigma_rad[i_clst]);
            path_AoX_rad += rad(clst_AoX[i_clst]) * Q[i_clst] * e_fac;
        }
        path_AoX = deg(path_AoX_rad);

        // Calculate composite AS
        double val = 0.0;
        for (size_t i_clst = 0; i_clst < n_clst; ++i_clst)
        {
            double sig = sigma_rad[i_clst];
            double phi = rad(clst_AoX[i_clst]);
            double e_fac = std::exp(-4.442882938158366 / sig);

            double pq = 0.25 * Q[i_clst];
            double sq = sqrt2 / sig;

            double z = sq * (phi - path_AoX_rad);
            val += pq * sig * sig * (z * z - 2.0 * z + 2.0);

            z = sq * (phi - path_AoX_rad - pi);
            val -= pq * sig * sig * e_fac * (z * z - 2.0 * z + 2.0);

            z = -sq * (phi - path_AoX_rad + pi);
            val -= pq * sig * sig * e_fac * (z * z - 2.0 * z + 2.0);

            z = -sq * (phi - path_AoX_rad);
            val += pq * sig * sig * (z * z - 2.0 * z + 2.0);
        }
        path_AS = deg(std::sqrt(val));
    };

    // Compute MIMO correlation matrices
    // See: IEEE 802.11-03/940r4, Sec. 3. MIMO Matrix Formulation
    auto get_MIMO_correlation = [&](size_t n_clst,             // Number of clusters
                                    size_t n_ant,              // Number of antennas
                                    const double *clst_P,      // Per-cluster-power, linear scale, length n_clst
                                    const double *clst_AoX,    // Per-cluster angle, degree, length n_clst
                                    const double *clst_AS,     // Per-cluster AS, degree, length n_clst
                                    const double *antenna_pos, // ULA antenna positions in [lambda], length n_ant
                                    std::complex<double> *res) // Output: MIMO correlation matrix, col-major, length n_ant^2
    {
        // Map cluster AS to laplacian PAS using the provided lookup tables
        std::vector<double> sigma_rad(n_clst), Q(n_clst);
        laplacian_mapping(n_clst, clst_P, clst_AS, sigma_rad.data(), Q.data());

        // Initialize
        arma::vec D(n_ant), Rxx(n_ant), Rxy(n_ant);
        for (size_t i_ant = 0; i_ant < n_ant; ++i_ant)
        {
            D[i_ant] = 2.0 * pi * antenna_pos[i_ant];
            Rxx[i_ant] = std::cyl_bessel_j(0.0, D[i_ant]);
            Rxy[i_ant] = 0.0;
        }

        // Add paths contributions
        for (size_t i_clst = 0; i_clst < n_clst; ++i_clst)
        {
            // Common variables
            double sig = sigma_rad[i_clst];
            double phi = rad(clst_AoX[i_clst]);
            double pw = Q[i_clst];

            double s_over = sqrt2 / sig;
            double e_fac = std::exp(-4.442882938158366 / sigma_rad[i_clst]);

            // even orders:  sin(2m*pi)=0, cos(2m*pi)=1  -> (sqrt2/sig)*(1 - e_fac)
            // odd  orders:  sin((2m+1)pi)=0, cos((2m+1)pi)=-1 -> (sqrt2/sig)*(1 + e_fac)
            double bracket_xx = s_over * (1.0 - e_fac);
            double bracket_xy = s_over * (1.0 + e_fac);

            // even orders n = 2,4,...,200
            for (int m = 1; m <= 100; ++m)
            {
                double dm = (double)m;
                double n = 2.0 * dm;
                double den = 4.0 * sqrt2 * sig * dm * dm + (sqrt8 / sig);
                double scale = 4.0 * std::cos(n * phi) * bracket_xx / den;
                for (size_t i_ant = 0; i_ant < n_ant; ++i_ant)
                    Rxx[i_ant] += pw * scale * std::cyl_bessel_j(n, D[i_ant]);
            }

            // odd orders n = 1,3,...,201  (include the n=201 term)
            for (int m = 0; m <= 100; ++m)
            {
                double dm = (double)m;
                double n = 2.0 * dm + 1.0;
                double den = sqrt2 * sig * (s_over * s_over + n * n);
                double scale = 4.0 * std::sin(n * phi) * bracket_xy / den;
                for (size_t i_ant = 0; i_ant < n_ant; ++i_ant)
                    Rxy[i_ant] += pw * scale * std::cyl_bessel_j(n, D[i_ant]);
            }
        }

        // Calculate Toeplitz matrix
        hermitian_toeplitz_matrix(n_ant, Rxx.memptr(), Rxy.memptr(), res);
    };

    // Computation of the Rice steering matrix assuming a general ULA with arbitrary normalized positions
    auto get_Rice_matrix = [&](size_t n_rx,                  // Number of RX antennas
                               size_t n_tx,                  // Number of TX antennas
                               const double *antenna_pos_Rx, // RX ULA antenna positions in [lambda], length n_rx
                               const double *antenna_pos_Tx, // TX ULA antenna positions in [lambda], length n_tx
                               double AoD_LOS_Rx_deg,        // Angle of departure of the LOS component at TX in degree
                               double AoD_LOS_Tx_deg,        // Angle of arrival of the LOS component at RX in degree
                               std::complex<double> *res)    // Output: Rice steering matrix, col-major, length n_rx * n_tx
    {
        // a_tx[m] = e^{j 2π pos_Tx[m] sin(theta_tx)}
        // a_rx[n] = e^{j 2π pos_Rx[n] sin(theta_rx)}
        // Rice matrix = a_rx * a_tx^T  (outer product)
        for (size_t i_tx = 0; i_tx < n_tx; ++i_tx)
        {
            double phase_tx = 2.0 * pi * antenna_pos_Tx[i_tx] * std::sin(rad(AoD_LOS_Tx_deg));
            std::complex<double> a_tx = std::polar(1.0, phase_tx);

            for (size_t i_rx = 0; i_rx < n_rx; ++i_rx)
            {
                double phase_rx = 2.0 * pi * antenna_pos_Rx[i_rx] * std::sin(rad(AoD_LOS_Rx_deg));
                std::complex<double> a_rx = std::polar(1.0, phase_rx);
                res[i_tx * n_rx + i_rx] = a_rx * a_tx; // column-major
            }
        }
    };

    // ---------- MODEL STEPS ----------

    // Get the number of paths
    arma::uword n_cluster = power_clst_dB.n_rows;
    arma::uword n_path = power_clst_dB.n_cols;

    // Convert path and cluster powers from dB to linear values
    arma::mat power_clst = power_clst_dB.transform(dB_2_linear);
    arma::mat power_avg = power_avg_dB.transform(dB_2_linear);

    // Calculate Pathloss for each user
    arma::vec Pathloss_dB(n_users, arma::fill::value(20.0 * std::log10(4.0 * pi * CarrierFreq_Hz / 3.0e8)));
    for (int i_user = 0; i_user < (int)n_users; ++i_user)
        if (Dist_m[i_user] < dBP_m)
            Pathloss_dB[i_user] += 20.0 * std::log10(Dist_m[i_user]);
        else
            Pathloss_dB[i_user] += 20.0 * std::log10(dBP_m) + 35.0 * std::log10(Dist_m[i_user] / dBP_m);

    // Set Shadow Fading STD for each user
    arma::vec Shadow_Fading_dB(n_users);
    for (int i_user = 0; i_user < (int)n_users; ++i_user)
        Shadow_Fading_dB[i_user] = (Dist_m[i_user] < dBP_m) ? 3.0 : SF_std_dB_NLOS;

    // Generate offset angles for MU-MIMO
    // See: TGac IEEE 802.11-09/0308r12, Appendix A – Generation of Pseudorandom Per-User AoA and AoD Offsets for MU-MIMO Channel Model
    // For 1 user, use TGn angles (AoD_offset = AoA_offset = 0.0) or supplied angles
    arma::vec AoD_LOS_offset_deg(n_users, arma::fill::value(AoD_offset));  // LOS offset angles @ TX
    arma::vec AoA_LOS_offset_deg(n_users, arma::fill::value(AoA_offset));  // LOS offset angles @ RX
    arma::vec AoD_NLOS_offset_deg(n_users, arma::fill::value(AoD_offset)); // NLOS offset angles @ TX
    arma::vec AoA_NLOS_offset_deg(n_users, arma::fill::value(AoA_offset)); // NLOS offset angles @ RX
    if (n_users != 1)
    {
        arma::uword seed = downlink ? 608341199 : 266639588;
        get_offset_angles(seed, n_users, AoD_LOS_offset_deg.memptr());

        seed = downlink ? 266639588 : 608341199;
        get_offset_angles(seed, n_users, AoA_LOS_offset_deg.memptr());

        seed = downlink ? 1468335517 : 115415752;
        get_offset_angles(seed, n_users, AoD_NLOS_offset_deg.memptr());

        seed = downlink ? 115415752 : 1468335517;
        get_offset_angles(seed, n_users, AoA_NLOS_offset_deg.memptr());
    }

    // User loop (only for TGac, TGn has only one user)
    for (int i_user = 0; i_user < (int)n_users; ++i_user)
    {
        // Apply offset AoD and AoA
        arma::mat AoD_clst_deg = AoD_deg;
        arma::mat AoA_clst_deg = AoA_deg;

        if (sdma_model == 1) // All clusters (default)
            AoD_clst_deg += AoD_NLOS_offset_deg[i_user], AoA_clst_deg += AoA_NLOS_offset_deg[i_user];
        else if (sdma_model == 2) // LOS only
            AoD_clst_deg.row(0) += AoD_NLOS_offset_deg[i_user], AoA_clst_deg.row(0) += AoA_NLOS_offset_deg[i_user];

        // Compute composite departure and arrival angles and angle spreads
        arma::vec AoD_composite(n_path), ASD_composite(n_path), AoA_composite(n_path), ASA_composite(n_path);
        for (arma::uword i_path = 0; i_path < n_path; ++i_path)
        {
            get_composite_path_angles(n_cluster, power_clst.colptr(i_path), AoD_clst_deg.colptr(i_path), ASD_deg.colptr(i_path), AoD_composite[i_path], ASD_composite[i_path]);
            get_composite_path_angles(n_cluster, power_clst.colptr(i_path), AoA_clst_deg.colptr(i_path), ASA_deg.colptr(i_path), AoA_composite[i_path], ASA_composite[i_path]);
        }

        // Computation of the power delay profile of the (LOS+NLOS) power
        arma::vec pdp_linear(n_path, arma::fill::ones);              // Initialize with all ones
        pdp_linear[0] += (Dist_m[i_user] < dBP_m) ? KF_linear : 0.0; // Increase the LOS-power for small distances below break-point
        pdp_linear %= power_avg;                                     // Element-wise multiply with composite path powers
        pdp_linear /= arma::accu(pdp_linear);                        // Normalize to unit-sum-power

        // Computation of the Rice steering matrix for the LOS path
        arma::vec antenna_pos_Tx = arma::linspace(0.0, (double)(n_tx - 1) * Tx_spacing, n_tx);
        arma::vec antenna_pos_Rx = arma::linspace(0.0, (double)(n_rx - 1) * Rx_spacing, n_rx);
        arma::cx_vec RiceMatrix(n_rx * n_tx); // Vectorized, col-major
        {
            double tx_deg = AoD_deg[0] + AoD_LOS_offset_deg[i_user];
            double rx_deg = AoA_deg[0] + AoA_LOS_offset_deg[i_user];
            get_Rice_matrix(n_rx, n_tx, antenna_pos_Rx.memptr(), antenna_pos_Tx.memptr(), rx_deg, tx_deg, RiceMatrix.memptr());
        }

        // Calculate MIMO correlation matrices at the TX and RX for each path
        arma::cx_cube RTx(n_tx, n_tx, n_path), RRx(n_rx, n_rx, n_path);
        for (arma::uword i_path = 0; i_path < n_path; ++i_path)
        {
            get_MIMO_correlation(n_cluster, n_tx, power_clst.colptr(i_path), AoD_clst_deg.colptr(i_path), ASD_deg.colptr(i_path), antenna_pos_Tx.memptr(), RTx.slice_memptr(i_path));
            get_MIMO_correlation(n_cluster, n_rx, power_clst.colptr(i_path), AoA_clst_deg.colptr(i_path), ASA_deg.colptr(i_path), antenna_pos_Rx.memptr(), RRx.slice_memptr(i_path));
        }

        // Generate channel matrices for each path
        arma::cx_cube H(n_rx, n_tx, n_path);
        for (arma::uword i_path = 0; i_path < n_path; ++i_path)
        {
            // Spatial correlation shaping matrix by Cholesky decomposition of the Kronecker product of (RTx, RRx)
            arma::cx_mat C = arma::chol(arma::kron(RTx.slice(i_path), RRx.slice(i_path))).t();

            // Initialize with IID coefficients
            arma::cx_vec h = arma::randn<arma::cx_vec>(n_rx * n_tx) * isqrt2;

            // Apply MIMO correlation
            h = C * h;

            // Apply path power
            h *= std::sqrt(pdp_linear[i_path]);

            // Add Rice matrix to LOS path
            if (i_path == 0 && Dist_m[i_user] < dBP_m)
            {
                double k_LOS = std::sqrt(KF_linear / (KF_linear + 1.0));
                double k_NLOS = std::sqrt(1.0 / (KF_linear + 1.0));
                h = RiceMatrix * std::sqrt(pdp_linear[0]) * k_LOS + h * k_NLOS;
            }

            // Store output
            std::memcpy(H.slice_memptr(i_path), h.memptr(), n_rx * n_tx * sizeof(std::complex<double>));
        }
    }

    return;
}

#endif