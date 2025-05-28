function test_get_channels_spherical

% Generate test antenna
ant = quadriga_lib.arrayant_generate('omni');
ant.e_theta_re(:,:,2) = 2;
ant.e_theta_im(:,:,2) = 0;
ant.e_phi_re(:,:,2) = 0;
ant.e_phi_im(:,:,2) = 0;
ant.element_pos = [0,0; 1,-1; 0,0];
ant.coupling_re = eye(2);
ant.coupling_im = zeros(2);

fbs_pos = [10,0 ; 0,10 ; 1 11];
path_gain = [1,0.25];
path_length = [0,0];
M = [1,1 ; 0,0 ; 0,0 ; 0,0 ; 0,0 ; 0,0 ; -1,-1 ; 0,0];

[coeff_re, coeff_im, delay, aod, eod, aoa, eoa] = quadriga_lib.get_channels_spherical( ant, ant, ...
    fbs_pos, fbs_pos, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1, 0 );

aod = aod * 180/pi;
eod = eod * 180/pi;
aoa = aoa * 180/pi;
eoa = eoa * 180/pi;

alpha = atan(2.0 / 20.0) * 180.0 / pi;
beta = 90.0;
assertElementsAlmostEqual( aod(:,:,1), [0,alpha;-alpha,0], 'absolute', 1e-14 );
assertElementsAlmostEqual( aod(:,:,2), [beta,beta;beta,beta], 'absolute', 1e-14 );

alpha = 180 - alpha;
beta = 180;
assertElementsAlmostEqual( cosd(aoa(:,:,1)), cosd([beta,-alpha;alpha,beta]), 'absolute', 1e-14 );

alpha = 180.0 - atan(9.0 / 20.0) * 180.0 / pi;
beta = 180.0 - atan(11.0 / 20.0) * 180.0 / pi;
assertElementsAlmostEqual( aoa(:,:,2), [alpha,alpha;beta,beta], 'absolute', 1e-14 );

alpha = atan(10.0 / 11.0) * 180.0 / pi;
beta = atan(10.0 / 9.0) * 180.0 / pi;
assertElementsAlmostEqual( eod(:,:,1), [0,0;0,0], 'absolute', 1e-14 );
assertElementsAlmostEqual( eod(:,:,2), [beta,alpha;beta,alpha], 'absolute', 1e-13 );

alpha = atan(10.0 / sqrt(9.0 * 9.0 + 20.0 * 20.0)) * 180.0 / pi;
beta = atan(10.0 / sqrt(11.0 * 11.0 + 20.0 * 20.0)) * 180.0 / pi;
assertElementsAlmostEqual( eoa(:,:,1), [0,0;0,0], 'absolute', 1e-14 );
assertElementsAlmostEqual( eoa(:,:,2), [alpha,alpha;beta,beta], 'absolute', 1e-13 );

amp = coeff_re.^2 + coeff_im.^2;
assertElementsAlmostEqual( amp(:,:,1), [1,4;4,16], 'absolute', 1e-13 );
assertElementsAlmostEqual( amp(:,:,2), [0.25,1;1,4], 'absolute', 1e-13 );

C = 299792458.0;
d0 = 20.0;
d1 = sqrt(20.0 * 20.0 + 2.0 * 2.0);
e0 = (sqrt(9.0 * 9.0 + 10.0 * 10.0) + sqrt(9.0 * 9.0 + 20.0 * 20.0 + 10.0 * 10.0));
e1 = (sqrt(9.0 * 9.0 + 10.0 * 10.0) + sqrt(11.0 * 11.0 + 20.0 * 20.0 + 10.0 * 10.0));
e2 = (sqrt(11.0 * 11.0 + 10.0 * 10.0) + sqrt(9.0 * 9.0 + 20.0 * 20.0 + 10.0 * 10.0));
e3 = (sqrt(11.0 * 11.0 + 10.0 * 10.0) + sqrt(11.0 * 11.0 + 20.0 * 20.0 + 10.0 * 10.0));

assertElementsAlmostEqual( delay(:,:,1), [d0,d1;d1,d0]/C, 'absolute', 1e-13 );
assertElementsAlmostEqual( delay(:,:,2), [e0,e2;e1,e3]/C, 'absolute', 1e-13 );

% Exception handling
[~] = quadriga_lib.get_channels_spherical( ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);

ant = rmfield(ant,'element_pos');
[~] = quadriga_lib.get_channels_spherical( ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);

ant = rmfield(ant,'coupling_re');
try
    [~] = quadriga_lib.get_channels_spherical( ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Transmit antenna: Imaginary part of coupling matrix (phase component) defined without real part (absolute component)';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

ant = rmfield(ant,'coupling_im');
[~] = quadriga_lib.get_channels_spherical( ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);

% Mismatching n_path
try
    [~] = quadriga_lib.get_channels_spherical( ant, ant, fbs_pos(:,[1,2,2]), fbs_pos, path_gain, path_length, M, [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Inputs ''fbs_pos'', ''lbs_pos'', ''path_gain'', ''path_length'', and ''M'' must have the same number of columns (n_paths)';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Wrong size of TX position
try
    [~] = quadriga_lib.get_channels_spherical( ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M, [0;0;1;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''tx_pos'' has incorrect number of elements.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end
