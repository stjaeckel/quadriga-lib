function test_calc_rician_k_factor()
% MOxUnit tests for quadriga_lib.calc_rician_k_factor

% --- Basic functionality ---
powers = [1.0, 0.5, 0.25]';  % [3, 1] - single snapshot with 3 paths
path_length = [10.0, 11.0, 12.0]';
tx_pos = [0; 0; 0];
rx_pos = [10; 0; 0];  % dTR = 10.0

[kf, pg] = quadriga_lib.calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, 0.01);

assertEqual(size(kf), [1, 1]);
assertEqual(size(pg), [1, 1]);
assertElementsAlmostEqual(kf, 1.0 / 0.75, 'absolute', 1e-10);
assertElementsAlmostEqual(pg, 1.75, 'absolute', 1e-10);

% --- Multiple paths within LOS window ---
powers2 = [2.0; 1.0; 0.5];
path_length2 = [10.0; 10.005; 15.0];

[kf2, pg2] = quadriga_lib.calc_rician_k_factor(powers2, path_length2, tx_pos, rx_pos, 0.01);

assertElementsAlmostEqual(kf2, 3.0 / 0.5, 'absolute', 1e-10);
assertElementsAlmostEqual(pg2, 3.5, 'absolute', 1e-10);

% --- No NLOS paths (infinite K-Factor) ---
powers3 = [1.0; 0.5];
path_length3 = [10.0; 10.005];

kf3 = quadriga_lib.calc_rician_k_factor(powers3, path_length3, tx_pos, rx_pos, 0.01);

assertEqual(isinf(kf3), true);
assertEqual(kf3 > 0, true);

% --- No LOS paths (zero K-Factor) ---
powers4 = [1.0; 0.5];
path_length4 = [11.0; 12.0];

[kf4, pg4] = quadriga_lib.calc_rician_k_factor(powers4, path_length4, tx_pos, rx_pos, 0.01);

assertEqual(kf4, 0.0);
assertElementsAlmostEqual(pg4, 1.5, 'absolute', 1e-10);

% --- Multiple snapshots with mobile RX ---
powers5 = [1.0, 2.0; 0.5, 1.0];  % [2, 2] - 2 paths, 2 snapshots
path_length5 = [10.0, 20.0; 12.0, 25.0];
rx_pos5 = [10.0, 20.0; 0.0, 0.0; 0.0, 0.0];

[kf5, pg5] = quadriga_lib.calc_rician_k_factor(powers5, path_length5, tx_pos, rx_pos5, 0.01);

assertEqual(size(kf5), [2, 1]);
assertEqual(size(pg5), [2, 1]);
assertElementsAlmostEqual(kf5(1), 1.0 / 0.5, 'absolute', 1e-10);
assertElementsAlmostEqual(kf5(2), 2.0 / 1.0, 'absolute', 1e-10);
assertElementsAlmostEqual(pg5(1), 1.5, 'absolute', 1e-10);
assertElementsAlmostEqual(pg5(2), 3.0, 'absolute', 1e-10);

% --- 3D positions ---
powers6 = [1.0; 0.5];
path_length6 = [5.0; 8.0];
rx_pos6 = [3; 4; 0];  % dTR = 5.0

kf6 = quadriga_lib.calc_rician_k_factor(powers6, path_length6, tx_pos, rx_pos6, 0.01);

assertElementsAlmostEqual(kf6, 1.0 / 0.5, 'absolute', 1e-10);

% --- Custom window size ---
powers7 = [1.0; 0.5; 0.25];
path_length7 = [10.0; 10.5; 12.0];

kf7a = quadriga_lib.calc_rician_k_factor(powers7, path_length7, tx_pos, rx_pos, 0.01);
assertElementsAlmostEqual(kf7a, 1.0 / 0.75, 'absolute', 1e-10);

kf7b = quadriga_lib.calc_rician_k_factor(powers7, path_length7, tx_pos, rx_pos, 1.0);
assertElementsAlmostEqual(kf7b, 1.5 / 0.25, 'absolute', 1e-10);

% --- Default window size (omit 5th argument) ---
kf8 = quadriga_lib.calc_rician_k_factor(powers, path_length, tx_pos, rx_pos);
assertElementsAlmostEqual(kf8, 1.0 / 0.75, 'absolute', 1e-10);

% --- Error: wrong number of input arguments ---
try
    quadriga_lib.calc_rician_k_factor(powers, path_length, tx_pos);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% --- Error: wrong tx_pos shape ---
try
    quadriga_lib.calc_rician_k_factor(powers, path_length, [0; 0], rx_pos, 0.01);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'tx_pos';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end
