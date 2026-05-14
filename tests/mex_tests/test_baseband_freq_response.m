function test_baseband_freq_response

% ---- Common single-frequency test data --------------------------------
% [n_rx=4, n_tx=3, n_path=2]
coeff = zeros(4,3,2);
coeff(:,1,1) = 0.25:0.25:1;
coeff(:,2,1) = 1:4;
coeff(:,3,1) = 1j*(1:4);
coeff(:,:,2) = -coeff(:,:,1);

fc = 299792458.0;   % wavelength = 1 m

% Planar-wave delays, broadcast across all RX/TX: [1, 1, n_path]
delay = zeros(1,1,2);
delay(1) = 1.0/fc;
delay(2) = 1.5/fc;

pilots = 0:0.1:2;   % 21 pilot positions, range 0..2

% ---- Test 1: legacy API (pilot_grid + bandwidth) ----------------------
[ hmat_re, hmat_im ] = quadriga_lib.baseband_freq_response( real(coeff), imag(coeff), delay, pilots, fc );

assertEqual( size(hmat_re), [4 3 21] );
assertEqual( size(hmat_im), [4 3 21] );

% pilot=0 and pilot=2 -> rotation multiple of 2pi; slice0 + slice1 = 0
T = zeros(4, 3);
assertElementsAlmostEqual( hmat_re(:,:,1),  T, 'absolute', 1.5e-6 );
assertElementsAlmostEqual( hmat_im(:,:,1),  T, 'absolute', 1.5e-6 );
assertElementsAlmostEqual( hmat_re(:,:,21), T, 'absolute', 3e-6 );
assertElementsAlmostEqual( hmat_im(:,:,21), T, 'absolute', 3e-6 );

% pilot=1 -> slice0 phase = 2pi (=> 1), slice1 phase = 3pi (=> -1), H = 2*slice0
T = [ 0.5, 2, 0 ; 1, 4, 0 ; 1.5, 6, 0 ; 2, 8, 0 ];
assertElementsAlmostEqual( hmat_re(:,:,11), T, 'absolute', 1.5e-6 );

% Linearity: col 2 of coeff is 4x col 1
assertElementsAlmostEqual( hmat_re(:,1,:)*4, hmat_re(:,2,:), 'absolute', 1.5e-6 );

% ---- Test 2: new API (center_freq + carrier_freq) ---------------------
% Wrapper derives: bw = carrier(end)-carrier(1), pilot = (carrier-carrier(1))/bw
% Choosing carrier_freq = anything + pilots*fc makes pilot*bw = pilots*fc -> same H
center_freq  = 28e9;                       % ignored under this derivation
carrier_freq = center_freq + (pilots(:))*fc;

[ hmat_re2, hmat_im2 ] = quadriga_lib.baseband_freq_response( real(coeff), imag(coeff), delay, [], [], center_freq, carrier_freq );

assertEqual( size(hmat_re2), [4 3 21] );
assertElementsAlmostEqual( hmat_re2, hmat_re, 'absolute', 3e-6 );
assertElementsAlmostEqual( hmat_im2, hmat_im, 'absolute', 3e-6 );

% ---- Test 3: center_freq alongside pilot+bw is silently ignored -------
[ hmat_re3, hmat_im3 ] = quadriga_lib.baseband_freq_response( real(coeff), imag(coeff), delay, pilots, fc, 28e9 );
assertElementsAlmostEqual( hmat_re3, hmat_re, 'absolute', 1e-12 );
assertElementsAlmostEqual( hmat_im3, hmat_im, 'absolute', 1e-12 );

% ---- Common multi-frequency setup -------------------------------------
% Replicate envelope (= coeff) across 3 input frequencies and bake in the
% delay phase exp(-j 2pi freq_in tau), simulating get_channels_multifreq output.
freq_in  = [1e9 ; 2e9 ; 3e9];

coeff_re_4d = zeros(4, 3, 2, 3);
coeff_im_4d = zeros(4, 3, 2, 3);
delay_4d    = repmat(delay, [1 1 1 3]);  % [1, 1, 2, 3]
for f = 1:3
    for p = 1:2
        rot = exp(-1j*2*pi*freq_in(f)*delay(1,1,p));
        c   = (real(coeff(:,:,p)) + 1j*imag(coeff(:,:,p))) * rot;
        coeff_re_4d(:,:,p,f) = real(c);
        coeff_im_4d(:,:,p,f) = imag(c);
    end
end

% ---- Test 3b: single carrier via pilot + bandwidth -------------------
% pilot=0.5, bw=fc -> pilot*bw = 0.5*fc. Must match slice from a multi-carrier call.
[ hmat_re_s1, hmat_im_s1 ] = quadriga_lib.baseband_freq_response( real(coeff), imag(coeff), delay, 0.5, fc );

assertEqual( size(hmat_re_s1), [4 3] );
assertEqual( size(hmat_im_s1), [4 3] );

pilots_ref = [0.4, 0.5, 0.6];
[ hmat_re_ref, hmat_im_ref ] = quadriga_lib.baseband_freq_response( real(coeff), imag(coeff), delay, pilots_ref, fc );

assertElementsAlmostEqual( hmat_re_s1, hmat_re_ref(:,:,2), 'absolute', 1e-6 );
assertElementsAlmostEqual( hmat_im_s1, hmat_im_ref(:,:,2), 'absolute', 1e-6 );

% ---- Test 3c: single carrier via center_freq + carrier_freq ----------
% Wrapper takes the absolute offset (carrier-center) path because span=0.
% Pick offset = 0.5*fc -> must equal the pilot=0.5, bw=fc result above.
center_freq_s  = 28e9;
carrier_freq_s = center_freq_s + 0.5*fc;

[ hmat_re_s2, hmat_im_s2 ] = quadriga_lib.baseband_freq_response( real(coeff), imag(coeff), delay, [], [], center_freq_s, carrier_freq_s );

assertEqual( size(hmat_re_s2), [4 3] );
assertEqual( size(hmat_im_s2), [4 3] );
assertElementsAlmostEqual( hmat_re_s2, hmat_re_s1, 'absolute', 1e-6 );
assertElementsAlmostEqual( hmat_im_s2, hmat_im_s1, 'absolute', 1e-6 );

% Degenerate: carrier == center -> pilot=0, expected = sum of path-0 slice
% (slice 1 has tau=1.5/fc, but pilot*bw=0 means zero rotation, so H = coeff(:,:,1) + coeff(:,:,2) = 0)
[ hmat_re_s3, hmat_im_s3 ] = quadriga_lib.baseband_freq_response( real(coeff), imag(coeff), delay, [], [], center_freq_s, center_freq_s );
T = zeros(4, 3);
assertElementsAlmostEqual( hmat_re_s3, T, 'absolute', 1e-6 );
assertElementsAlmostEqual( hmat_im_s3, T, 'absolute', 1e-6 );

% ---- Test 4: multi-freq output, cross-check against single-freq ------
% Constant envelope across freq_in -> SLERP is identity. Multi-freq H must
% match a single-freq call with pilot*bandwidth = absolute freq_out.
freq_out = linspace(1e9, 3e9, 11)';

[ hmat_re_m,  hmat_im_m  ] = quadriga_lib.baseband_freq_response( coeff_re_4d, coeff_im_4d, delay_4d, [], [], freq_in, freq_out );
[ hmat_re_sf, hmat_im_sf ] = quadriga_lib.baseband_freq_response( real(coeff), imag(coeff), delay, freq_out/fc, fc );

assertEqual( size(hmat_re_m), [4 3 11] );
assertElementsAlmostEqual( hmat_re_m, hmat_re_sf, 'absolute', 1e-4 );
assertElementsAlmostEqual( hmat_im_m, hmat_im_sf, 'absolute', 1e-4 );

% ---- Test 5: multi-freq via derived carrier (pilot+bw+center) --------
% carrier = center_freq(1) + pilot*bw  must reproduce freq_out
pilot_d = (freq_out - freq_in(1)) / (freq_out(end) - freq_in(1));
bw_d    = freq_out(end) - freq_in(1);

[ hmat_re_d, hmat_im_d ] = quadriga_lib.baseband_freq_response( coeff_re_4d, coeff_im_4d, delay_4d, pilot_d, bw_d, freq_in );

assertElementsAlmostEqual( hmat_re_d, hmat_re_m, 'absolute', 1e-6 );
assertElementsAlmostEqual( hmat_im_d, hmat_im_m, 'absolute', 1e-6 );

% ---- Test 6: SLERP magnitude interpolation across freq_in ------------
% Envelope amplitude ramps 1, 2, 3 across freq_in; expect linear interp.
ramp_re = zeros(1, 1, 1, 3);
ramp_im = zeros(1, 1, 1, 3);
ramp_dl = zeros(1, 1, 1, 3);
amps  = [1 2 3];
tau_r = 100e-9;
for f = 1:3
    rot = exp(-1j*2*pi*freq_in(f)*tau_r);
    c   = amps(f) * rot;
    ramp_re(1,1,1,f) = real(c);
    ramp_im(1,1,1,f) = imag(c);
    ramp_dl(1,1,1,f) = tau_r;
end

[ hmat_re_r, hmat_im_r ] = quadriga_lib.baseband_freq_response( ramp_re, ramp_im, ramp_dl, [], [], freq_in, freq_out );

mags = squeeze( sqrt(hmat_re_r.^2 + hmat_im_r.^2) );
expected_mags = interp1(freq_in, amps, freq_out);
assertElementsAlmostEqual( mags, expected_mags, 'absolute', 1e-5 );

% ---- Test 4b: multi-freq with a single output carrier ----------------
% Same setup as Test 4 (constant envelope -> SLERP is identity).
% Multi-freq H at one carrier must match single-freq H at pilot*bw = carrier.
fx = 1.5e9;
[ hmat_re_m1, hmat_im_m1 ] = quadriga_lib.baseband_freq_response( coeff_re_4d, coeff_im_4d, delay_4d, [], [], freq_in, fx );
[ hmat_re_r1, hmat_im_r1 ] = quadriga_lib.baseband_freq_response( real(coeff), imag(coeff), delay, fx/fc, fc );

assertEqual( size(hmat_re_m1), [4 3] );
assertEqual( size(hmat_im_m1), [4 3] );
assertElementsAlmostEqual( hmat_re_m1, hmat_re_r1, 'absolute', 1e-4 );
assertElementsAlmostEqual( hmat_im_m1, hmat_im_r1, 'absolute', 1e-4 );

% Same via derived carrier (pilot+bw+center, single pilot)
[ hmat_re_m2, hmat_im_m2 ] = quadriga_lib.baseband_freq_response( coeff_re_4d, coeff_im_4d, delay_4d, (fx-freq_in(1))/fc, fc, freq_in );

assertEqual( size(hmat_re_m2), [4 3] );
assertElementsAlmostEqual( hmat_re_m2, hmat_re_m1, 'absolute', 1e-6 );
assertElementsAlmostEqual( hmat_im_m2, hmat_im_m1, 'absolute', 1e-6 );

% ---- Error: 3D coeff with no carrier source --------------------------
try
    quadriga_lib.baseband_freq_response( real(coeff), imag(coeff), delay, [], [] );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Provide pilot_grid+bandwidth or center_freq+carrier_freq';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% ---- Error: 4D coeff with center_freq but no carrier source ----------
try
    quadriga_lib.baseband_freq_response( coeff_re_4d, coeff_im_4d, delay_4d, [], [], freq_in );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Provide pilot_grid+bandwidth or carrier_freq';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end


% ---- Error: all four carrier-spec inputs given ------------------------
try
    quadriga_lib.baseband_freq_response( real(coeff), imag(coeff), delay, pilots, fc, 28e9, carrier_freq );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Specify either';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% ---- Error: multi-freq input without center_freq ----------------------
try
    quadriga_lib.baseband_freq_response( coeff_re_4d, coeff_im_4d, delay_4d, pilots, fc );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'center_freq is required';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% ---- Error: center_freq length does not match 4th dim of coeff -------
try
    quadriga_lib.baseband_freq_response( coeff_re_4d, coeff_im_4d, delay_4d, [], [], [1e9; 2e9], freq_out );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Length of center_freq';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% ---- Error: too many output arguments ---------------------------------
try
    [~,~,~] = quadriga_lib.baseband_freq_response( real(coeff), imag(coeff), delay, pilots, fc );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% ---- Error: too many input arguments ---------------------------------
try
    quadriga_lib.baseband_freq_response( real(coeff), imag(coeff), delay, pilots, fc, 28e9, carrier_freq, true );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end
