function test_get_channels_planar

% Generate test antenna
ant = quadriga_lib.arrayant_generate('omni');
ant.e_theta_re(:,:,2) = 2;
ant.e_theta_im(:,:,2) = 0;
ant.e_phi_re(:,:,2) = 0;
ant.e_phi_im(:,:,2) = 0;
ant.element_pos = [0,0; 1,-1; 0,0];
ant.coupling_re = eye(2);
ant.coupling_im = zeros(2);

aod = [0, 90] * pi/180;
eod = [0, 45] * pi/180;
aoa = [180, 180 - atand(10/20)] * pi/180;
eoa = [0, atand(10 / sqrt(10^2 + 20^2) )] * pi/180;

path_gain = [1,0.25];
path_length = [20,  sqrt(10^2 + 10^2) + sqrt(10^2 + 20^2 + 10^2)  ];
M = [1,1 ; 0,0 ; 0,0 ; 0,0 ; 0,0 ; 0,0 ; -1,-1 ; 0,0];

[coeff_re, coeff_im, delay, rx_Doppler] = quadriga_lib.get_channels_planar( ant, ant, ...
    aod, eod, aoa, eoa, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);

amp = coeff_re.^2 + coeff_im.^2;
assertElementsAlmostEqual( amp(:,:,1), [1,4;4,16], 'absolute', 1e-13 );
assertElementsAlmostEqual( amp(:,:,2), [0.25,1;1,4], 'absolute', 1e-13 );

C = 299792458.0;
d0 = 20.0;
d1 = 20.0;
e0 = (sqrt(9.0 * 9.0 + 10.0 * 10.0) + sqrt(9.0 * 9.0 + 20.0 * 20.0 + 10.0 * 10.0));
e1 = (sqrt(9.0 * 9.0 + 10.0 * 10.0) + sqrt(11.0 * 11.0 + 20.0 * 20.0 + 10.0 * 10.0));
e2 = (sqrt(11.0 * 11.0 + 10.0 * 10.0) + sqrt(9.0 * 9.0 + 20.0 * 20.0 + 10.0 * 10.0));
e3 = (sqrt(11.0 * 11.0 + 10.0 * 10.0) + sqrt(11.0 * 11.0 + 20.0 * 20.0 + 10.0 * 10.0));

assertElementsAlmostEqual( delay(:,:,1), [d0,d1;d1,d0]/C, 'absolute', 1e-13 );
assertElementsAlmostEqual( delay(:,:,2), [e0,e2;e1,e3]/C, 'absolute', 1.2e-10 );

Doppler = cos(aoa(2)) * cos(eoa(2));
assertElementsAlmostEqual( rx_Doppler, [-1, Doppler], 'absolute', 1e-13 );

% Exception handling
[~] = quadriga_lib.get_channels_planar( ant, ant, aod, eod, aoa, eoa, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);

ant = rmfield(ant,'element_pos');
[~] = quadriga_lib.get_channels_planar( ant, ant, aod, eod, aoa, eoa, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);

ant = rmfield(ant,'coupling_re');
try
    [~] = quadriga_lib.get_channels_planar( ant, ant, aod, eod, aoa, eoa, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Transmit antenna: Imaginary part of coupling matrix (phase component) defined without real part (absolute component)';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

ant = rmfield(ant,'coupling_im');
[~] = quadriga_lib.get_channels_planar( ant, ant, aod, eod, aoa, eoa, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);

% Mismatching n_path
try
    [~] = quadriga_lib.get_channels_planar( ant, ant, aod([1,2,2]), eod, aoa, eoa, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Inputs ''aod'', ''eod'', ''aoa'', ''eoa'', ''path_gain'', ''path_length'', and ''M'' must have the same number of columns (n_paths)';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Wrong size of TX position
try
    [~] = quadriga_lib.get_channels_planar( ant, ant, aod, eod, aoa, eoa, path_gain, path_length, M, [0;0;1;2], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''tx_pos'' has incorrect number of elements.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end
