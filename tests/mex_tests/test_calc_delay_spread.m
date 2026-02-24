% SPDX-License-Identifier: Apache-2.0
%
% quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
% Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
% http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
% ------------------------------------------------------------------------

function test_calc_delay_spread()
% MOxUnit tests for quadriga_lib.calc_delay_spread

% --- Basic single CIR with 3 paths ---
delays = [0, 1e-6, 2e-6];
powers = [1.0, 0.5, 0.25];

[ds, mean_delay] = quadriga_lib.calc_delay_spread( delays, powers );

assertEqual( size(ds), [1, 1] );
assertEqual( size(mean_delay), [1, 1] );

% Expected mean delay: (1.0*0 + 0.5*1e-6 + 0.25*2e-6) / 1.75
expected_mean = 1e-6 / 1.75;
assertElementsAlmostEqual( mean_delay, expected_mean, 'absolute', 1e-14 );

% Delay spread must be positive and less than max delay
assertTrue( ds > 0 );
assertTrue( ds < 2e-6 );

% --- Single path yields zero spread ---
delays = 5e-6;
powers = 1.0;

[ds, mean_delay] = quadriga_lib.calc_delay_spread( delays, powers );
assertElementsAlmostEqual( ds, 0.0, 'absolute', 1e-20 );
assertElementsAlmostEqual( mean_delay, 5e-6, 'absolute', 1e-20 );

% --- Two equal-power paths ---
delays = [0, 2e-6];
powers = [1.0, 1.0];

[ds, mean_delay] = quadriga_lib.calc_delay_spread( delays, powers );
assertElementsAlmostEqual( mean_delay, 1e-6, 'absolute', 1e-14 );
assertElementsAlmostEqual( ds, 1e-6, 'absolute', 1e-14 );

% --- Multiple CIRs (2 rows) ---
delays = [0, 1e-6, 0; 0, 1e-6, 2e-6];
powers = [1.0, 1.0, 0; 1.0, 1.0, 1.0];

[ds, mean_delay] = quadriga_lib.calc_delay_spread( delays, powers );

assertEqual( size(ds), [2, 1] );
assertEqual( size(mean_delay), [2, 1] );

% CIR 1: mean = 0.5e-6, DS = 0.5e-6 (third path has zero power)
assertElementsAlmostEqual( mean_delay(1), 0.5e-6, 'absolute', 1e-14 );
assertElementsAlmostEqual( ds(1), 0.5e-6, 'absolute', 1e-14 );

% CIR 2: mean = 1e-6
assertElementsAlmostEqual( mean_delay(2), 1e-6, 'absolute', 1e-14 );
assertTrue( ds(2) > 0 );

% --- Threshold filters weak paths ---
delays = [0, 10e-6, 1e-6];
powers = [1.0, 0.001, 0.5];

ds_20 = quadriga_lib.calc_delay_spread( delays, powers, 20.0 );
ds_all = quadriga_lib.calc_delay_spread( delays, powers, 100.0 );

% DS with all paths should be larger
assertTrue( ds_all > ds_20 );

% --- Granularity bins paths ---
delays = [100e-9, 110e-9, 1000e-9];
powers = [1.0, 1.0, 1.0];

ds_no_gran = quadriga_lib.calc_delay_spread( delays, powers, 100.0, 0.0 );
ds_gran = quadriga_lib.calc_delay_spread( delays, powers, 100.0, 50e-9 );

assertTrue( ds_no_gran > 0 );
assertTrue( ds_gran > 0 );

% --- Without mean_delay output ---
delays = [0, 1e-6];
powers = [1.0, 1.0];

ds = quadriga_lib.calc_delay_spread( delays, powers );
assertElementsAlmostEqual( ds, 0.5e-6, 'absolute', 1e-14 );

% --- Granularity with mean_delay output ---
delays = [0, 50e-9, 500e-9, 550e-9];
powers = [1.0, 1.0, 1.0, 1.0];

[ds, mean_delay] = quadriga_lib.calc_delay_spread( delays, powers, 100.0, 100e-9 );

assertTrue( mean_delay > 0 );
assertTrue( ds > 0 );

end
