function test_get_channels_multifreq

% ============================================================================
% Setup: 2-element omni antenna (mirrors test_get_channels_spherical geometry)
% ============================================================================
ant = quadriga_lib.arrayant_generate('omni');
ant.e_theta_re(:,:,2) = 2;
ant.e_theta_im(:,:,2) = 0;
ant.e_phi_re(:,:,2) = 0;
ant.e_phi_im(:,:,2) = 0;
ant.element_pos = [0,0; 1,-1; 0,0];
ant.coupling_re = eye(2);
ant.coupling_im = zeros(2);
ant.center_freq = 2997924580.0;     % required by arrayant_is_valid_multi

fbs_pos     = [10,0 ; 0,10 ; 1 11]; % 3 x n_path
path_length = [0,0];

% Single-frequency shapes (spherical reference)
pg_s = [1, 0.25];                                        % Col, n_path = 2
M_s  = [1,1; 0,0; 0,0; 0,0; 0,0; 0,0; -1,-1; 0,0];       % [8, n_path]

% Multi-frequency shapes (n_freq_in = 1)
pg_m = [1; 0.25];                                        % Mat [n_path, n_freq_in]
M_m  = reshape(M_s, [8, 2, 1]);                          % Cube [8, n_path, n_freq_in]

fc = 2997924580.0;

% ============================================================================
% Test 1: Multi-freq with single freq entry matches get_channels_spherical
%         + verify 4D output shape [n_rx, n_tx, n_path, n_freq_out]
% ============================================================================
[cr_ref, ci_ref, dl_ref] = quadriga_lib.get_channels_spherical( ant, ant, ...
    fbs_pos, fbs_pos, pg_s, path_length, M_s, ...
    [0;0;1], [0,0,0], [20;0;1], [0,0,0], fc, 1, 0 );

[cr, ci, dl] = quadriga_lib.get_channels_multifreq( ant, ant, ...
    fbs_pos, fbs_pos, pg_m, path_length, M_m, ...
    [0;0;1], [0,0,0], [20;0;1], [0,0,0], fc, fc, 1, 0 );

assertEqual( size(cr,1), 2 );   % n_rx
assertEqual( size(cr,2), 2 );   % n_tx
assertEqual( size(cr,3), 2 );   % n_path
assertEqual( size(cr,4), 1 );   % n_freq_out
assertEqual( size(dl,4), 1 );

assertElementsAlmostEqual( cr(:,:,:,1), cr_ref, 'absolute', 5e-6 );
assertElementsAlmostEqual( ci(:,:,:,1), ci_ref, 'absolute', 5e-6 );
assertElementsAlmostEqual( dl(:,:,:,1), dl_ref, 'absolute', 1e-12 );

% ============================================================================
% Test 2: Multiple output frequencies — trailing dim = n_freq_out
%         Delays are geometry-only → identical across freq_out slabs
% ============================================================================
fo3 = [1e9; 2e9; 3e9];
[cr3, ~, dl3] = quadriga_lib.get_channels_multifreq( ant, ant, ...
    fbs_pos, fbs_pos, pg_m, path_length, M_m, ...
    [0;0;1], [0,0,0], [20;0;1], [0,0,0], fc, fo3, 1, 0 );
assertEqual( size(cr3), [2,2,2,3] );
assertEqual( size(dl3), [2,2,2,3] );
assertElementsAlmostEqual( dl3(:,:,:,1), dl3(:,:,:,2), 'absolute', 1e-14 );
assertElementsAlmostEqual( dl3(:,:,:,2), dl3(:,:,:,3), 'absolute', 1e-14 );

% ============================================================================
% Test 3: path_gain linearly interpolated across input frequencies
%         pg=[1,4] at fi=[1,2]GHz → at 1.5GHz gain=2.5, amp=sqrt(2.5)
% ============================================================================
ant_o = quadriga_lib.arrayant_generate('omni');
ant_o.center_freq = 1e9;
fbs1 = [10;0;0];
pg2  = [1.0, 4.0];                                  % [n_path=1, n_freq_in=2]
M2   = zeros(2,1,2); M2(1,1,1) = 1; M2(1,1,2) = 1;
[crp, cip, ~] = quadriga_lib.get_channels_multifreq( ant_o, ant_o, ...
    fbs1, fbs1, pg2, 10, M2, [0;0;0], [0,0,0], [10;0;0], [0,0,0], ...
    [1e9; 2e9], 1.5e9, 1, 0 );
amp = sqrt(crp.^2 + cip.^2);
assertElementsAlmostEqual( amp, sqrt(2.5), 'absolute', 0.01 );

% ============================================================================
% Test 4: Scalar M (2 rows) matches full M (8 rows) when only VV is set
% ============================================================================
M_full = zeros(8,1,1); M_full(1,1,1) = 0.8; M_full(2,1,1) = 0.3;
M_sca  = zeros(2,1,1); M_sca(1,1,1)  = 0.8; M_sca(2,1,1)  = 0.3;
[cr_f, ci_f] = quadriga_lib.get_channels_multifreq( ant_o, ant_o, ...
    fbs1, fbs1, 1, 10, M_full, [0;0;0], [0,0,0], [10;0;0], [0,0,0], 1e9, 1e9 );
[cr_s, ci_s] = quadriga_lib.get_channels_multifreq( ant_o, ant_o, ...
    fbs1, fbs1, 1, 10, M_sca,  [0;0;0], [0,0,0], [10;0;0], [0,0,0], 1e9, 1e9 );
assertElementsAlmostEqual( cr_f, cr_s, 'absolute', 1e-12 );
assertElementsAlmostEqual( ci_f, ci_s, 'absolute', 1e-12 );

% ============================================================================
% Test 5: Acoustic propagation_speed (= 343 m/s) for d = 5 m → delay = 5/343
% ============================================================================
c_s = 343.0;
M1   = zeros(2,1,1); M1(1,1,1) = 1;
fbs5 = [5;0;0];
[~, ~, dl_a] = quadriga_lib.get_channels_multifreq( ant_o, ant_o, ...
    fbs5, fbs5, 1, 5, M1, [0;0;0], [0,0,0], [5;0;0], [0,0,0], ...
    1000, 1000, 1, 0, c_s );
assertElementsAlmostEqual( dl_a(1,1,1,1), 5/c_s, 'absolute', 1e-12 );

% ============================================================================
% Test 6: add_fake_los_path adds one extra path slice
% ============================================================================
fbs_n = [5;5;0];
[cr_l, ~, dl_l] = quadriga_lib.get_channels_multifreq( ant_o, ant_o, ...
    fbs_n, fbs_n, 1, 15, M1, [0;0;0], [0,0,0], [10;0;0], [0,0,0], ...
    1e9, 1e9, 0, 1 );             % add_fake_los_path = true
assertEqual( size(cr_l,3), 2 );   % 1 real + 1 fake LOS
assertEqual( size(dl_l,3), 2 );

% ============================================================================
% Test 7: Empty [] for optional propagation_speed → default speed of light
% ============================================================================
[~, ~, dl_d] = quadriga_lib.get_channels_multifreq( ant_o, ant_o, ...
    fbs1, fbs1, 1, 10, M1, [0;0;0], [0,0,0], [10;0;0], [0,0,0], ...
    1e9, 1e9, 1, 0, [] );
assertElementsAlmostEqual( dl_d(1,1,1,1), 10/299792458, 'absolute', 1e-15 );

% ============================================================================
% Test 8: Multi-entry struct array (2 input frequencies)
% ============================================================================
ant_mf = repmat(ant_o, 1, 2);
ant_mf(2).center_freq = 2e9;
pgm = [1.0, 1.0];
Mm  = zeros(2,1,2); Mm(1,1,:) = 1;
[cr_mf, ~, ~] = quadriga_lib.get_channels_multifreq( ant_mf, ant_mf, ...
    fbs1, fbs1, pgm, 10, Mm, [0;0;0], [0,0,0], [10;0;0], [0,0,0], ...
    [1e9; 2e9], [1e9; 2e9], 1, 0 );
assertEqual( size(cr_mf), [1,1,1,2] );

% ============================================================================
% Exception handling
% ============================================================================

% Wrong size of tx_pos — caught by qd_mex_typecast_Col in the wrapper
try
    [~] = quadriga_lib.get_channels_multifreq( ant, ant, fbs_pos, fbs_pos, pg_m, ...
        path_length, M_m, [0;0;1;1], [0,0,0], [20;0;1], [0,0,0], fc, fc, 1, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''tx_pos'' has incorrect number of elements.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "', ME.message, '"']);
    end
end

% Mismatched n_path between fbs_pos and lbs_pos (C++ validation)
try
    [~] = quadriga_lib.get_channels_multifreq( ant, ant, fbs_pos(:,[1,2,2]), fbs_pos, ...
        pg_m, path_length, M_m, [0;0;1], [0,0,0], [20;0;1], [0,0,0], fc, fc, 1, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
        error('moxunit:exceptionNotRaised', 'Expected error for mismatched n_path');
    end
end

% Wrong M row count (neither 8 nor 2)
M_bad = zeros(4, 2, 1);
try
    [~] = quadriga_lib.get_channels_multifreq( ant, ant, fbs_pos, fbs_pos, pg_m, ...
        path_length, M_bad, [0;0;1], [0,0,0], [20;0;1], [0,0,0], fc, fc, 1, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
        error('moxunit:exceptionNotRaised', 'Expected error for wrong M row count');
    end
end

% Empty tx_array struct — caught by the wrapper's empty guard
try
    [~] = quadriga_lib.get_channels_multifreq( struct([]), ant, fbs_pos, fbs_pos, pg_m, ...
        path_length, M_m, [0;0;1], [0,0,0], [20;0;1], [0,0,0], fc, fc, 1, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'tx_array must not be empty.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "', ME.message, '"']);
    end
end

% Negative propagation_speed (C++ validation)
try
    [~] = quadriga_lib.get_channels_multifreq( ant_o, ant_o, fbs1, fbs1, 1, 10, M1, ...
        [0;0;0], [0,0,0], [10;0;0], [0,0,0], 1e9, 1e9, 1, 0, -1.0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
        error('moxunit:exceptionNotRaised', 'Expected error for negative propagation_speed');
    end
end

end