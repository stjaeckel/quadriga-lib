function test_calc_cross_polarization_ratio()
% MOxUnit tests for quadriga_lib.calc_cross_polarization_ratio

% --- Basic NLOS XPR ---
% 1 LOS + 2 NLOS paths, TX at origin, RX at (10,0,0)
pw = [1.0, 0.5, 0.5; ]';  % Actually column-per-CIR, but single CIR so just a column
% Rearrange: powers is [n_path, n_cir] = [3, 1]
powers = [1.0; 0.5; 0.5];

% M: [8, n_path, n_cir] = [8, 3, 1]
M = zeros(8, 3);
M(:,1) = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]; % LOS
M(:,2) = [0.9, 0.0, 0.1, 0.0, 0.1, 0.0, 0.8, 0.0];   % NLOS 1
M(:,3) = [0.7, 0.0, 0.3, 0.0, 0.2, 0.0, 0.6, 0.0];   % NLOS 2

pl = [10.0; 12.0; 15.0];
tx_pos = [0; 0; 0];
rx_pos = [10; 0; 0];

[xpr, pg] = quadriga_lib.calc_cross_polarization_ratio(powers, M, pl, tx_pos, rx_pos);

assertEqual(size(xpr), [1, 6]);
assertEqual(size(pg), [1, 1]);

% pg includes all paths
assertElementsAlmostEqual(pg, 3.225, 'absolute', 1e-10);

% V-XPR = 0.65 / 0.05 = 13.0
assertElementsAlmostEqual(xpr(1,2), 13.0, 'absolute', 1e-10);

% H-XPR = 0.50 / 0.025 = 20.0
assertElementsAlmostEqual(xpr(1,3), 20.0, 'absolute', 1e-10);

% Aggregate linear XPR = 1.15 / 0.075
assertElementsAlmostEqual(xpr(1,1), 1.15 / 0.075, 'absolute', 1e-10);

% --- Include LOS test ---
powers2 = [1.0; 0.5];
M2 = zeros(8, 2);
M2(:,1) = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0];
M2(:,2) = [0.8, 0.0, 0.2, 0.0, 0.2, 0.0, 0.7, 0.0];
pl2 = [10.0; 15.0];

% Without LOS
xpr_no = quadriga_lib.calc_cross_polarization_ratio(powers2, M2, pl2, tx_pos, rx_pos, false);
assertElementsAlmostEqual(xpr_no(1,2), 16.0, 'absolute', 1e-10);

% With LOS
xpr_yes = quadriga_lib.calc_cross_polarization_ratio(powers2, M2, pl2, tx_pos, rx_pos, true);
assertElementsAlmostEqual(xpr_yes(1,2), 66.0, 'absolute', 1e-10);

% --- Complex M elements ---
M3 = [0.8, 0.2, 0.1, -0.1, 0.05, 0.05, 0.7, -0.3]';
pl3 = 20.0;
pw3 = 1.0;

[xpr3, pg3] = quadriga_lib.calc_cross_polarization_ratio(pw3, M3, pl3, tx_pos, rx_pos);

abs2_vv = 0.8^2 + 0.2^2;
abs2_hv = 0.1^2 + 0.1^2;
abs2_vh = 0.05^2 + 0.05^2;
abs2_hh = 0.7^2 + 0.3^2;

assertElementsAlmostEqual(pg3, abs2_vv + abs2_hv + abs2_vh + abs2_hh, 'absolute', 1e-14);
assertElementsAlmostEqual(xpr3(1,2), abs2_vv / abs2_hv, 'absolute', 1e-10);
assertElementsAlmostEqual(xpr3(1,3), abs2_hh / abs2_vh, 'absolute', 1e-10);
assertElementsAlmostEqual(xpr3(1,1), (abs2_vv + abs2_hh) / (abs2_hv + abs2_vh), 'absolute', 1e-10);

% --- Window size effect ---
powers_w = [1.0; 0.8; 0.5];
M_w = zeros(8, 3);
M_w(:,1) = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0];
M_w(:,2) = [0.9, 0.0, 0.15, 0.0, 0.1, 0.0, 0.85, 0.0];
M_w(:,3) = [0.7, 0.0, 0.3, 0.0, 0.2, 0.0, 0.6, 0.0];
pl_w = [10.0; 10.005; 12.0];

% Large window excludes paths 1 and 2
xpr_w = quadriga_lib.calc_cross_polarization_ratio(powers_w, M_w, pl_w, tx_pos, rx_pos, false, 0.01);
assertElementsAlmostEqual(xpr_w(1,2), 0.49/0.09, 'absolute', 1e-10);

% --- Equal M elements give XPR = 1 in linear basis ---
M_eq = [0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0]';
xpr_eq = quadriga_lib.calc_cross_polarization_ratio(1.0, M_eq, 20.0, tx_pos, rx_pos);
assertElementsAlmostEqual(xpr_eq(1,1), 1.0, 'absolute', 1e-10);
assertElementsAlmostEqual(xpr_eq(1,2), 1.0, 'absolute', 1e-10);
assertElementsAlmostEqual(xpr_eq(1,3), 1.0, 'absolute', 1e-10);

% --- Multiple CIRs ---
powers_mc = [1.0, 0.8; 0.5, 0.4; 0.0, 0.0]; % [3, 2] zero-padded
M_mc = zeros(8, 3, 2);
M_mc(:,1,1) = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0];
M_mc(:,2,1) = [0.8, 0.0, 0.2, 0.0, 0.15, 0.0, 0.7, 0.0];
M_mc(:,1,2) = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0];
M_mc(:,2,2) = [0.6, 0.0, 0.3, 0.0, 0.25, 0.0, 0.5, 0.0];
pl_mc = [10.0, 10.0; 14.0, 13.0; 0.0, 0.0];

tx_mc = [0, 1; 0, 0; 0, 0];
rx_mc = [10, 11; 0, 0; 0, 0];

xpr_mc = quadriga_lib.calc_cross_polarization_ratio(powers_mc, M_mc, pl_mc, tx_mc, rx_mc);
assertEqual(size(xpr_mc), [2, 6]);

% CIR 1 V-XPR: 0.32/0.02 = 16
assertElementsAlmostEqual(xpr_mc(1,2), 0.32/0.02, 'absolute', 1e-10);

% --- Error handling: wrong number of input arguments ---
try
    quadriga_lib.calc_cross_polarization_ratio(powers);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end
