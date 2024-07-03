function test_baseband_freq_response

coeff = zeros(4,3,2);
coeff(:,1,1) = 0.25:0.25:1;
coeff(:,2,1) = 1:4;
coeff(:,3,1) = 1j*(1:4);
coeff(:,:,2) = -coeff(:,:,1);

coeff(:,:,:,2) = 2*coeff(:,:,:,1);
coeff(:,:,:,3) = 3*coeff(:,:,:,1);

fc = 299792458.0;

delay = zeros(1,1,2);
delay(1) = 1/fc;
delay(2) = 1.5/fc;

delay(:,:,:,2) = delay(:,:,:,1);
delay(:,:,:,3) = delay(:,:,:,1);

pilots = 0:0.1:2;

[ hmat_re, hmat_im ] = quadriga_lib.baseband_freq_response( real(coeff), imag(coeff), delay, pilots, fc );

T = zeros(4, 3);
assertElementsAlmostEqual( hmat_re(:,:,1,1), T, 'absolute', 1.5e-6 );
assertElementsAlmostEqual( hmat_re(:,:,21,1), T, 'absolute', 3e-6 );
assertElementsAlmostEqual( hmat_im(:,:,1,1), T, 'absolute', 1.5e-6 );
assertElementsAlmostEqual( hmat_im(:,:,21,1), T, 'absolute', 3e-6 );

assertElementsAlmostEqual( hmat_re(:,1,:)*4, hmat_re(:,2,:), 'absolute', 1.5e-6 );

T = [ 0.5, 2, 0 ; 1,4,0 ; 1.5,6,0 ; 2,8,0 ];
assertElementsAlmostEqual( hmat_re(:,:,11,1), T, 'absolute', 1.5e-6 );

assertElementsAlmostEqual( hmat_re(:,:,:,1)*2, hmat_re(:,:,:,2), 'absolute', 1.5e-6 );
assertElementsAlmostEqual( hmat_re(:,:,:,1)*3, hmat_re(:,:,:,3), 'absolute', 2e-6 );

[ hmat_re, hmat_im ] = quadriga_lib.baseband_freq_response( real(coeff), imag(coeff), delay, pilots, fc, [2,3,2,1] );

T = [0.25, 1, -1 ; 0.5, 2, -2 ; 0.75, 3, -3 ; 1, 4, -4 ];
assertElementsAlmostEqual( hmat_im(:,:,16,4), T, 'absolute', 1e-5 );

assertElementsAlmostEqual( hmat_re(:,:,:,4)*2, hmat_re(:,:,:,1), 'absolute', 1.5e-6 );
assertElementsAlmostEqual( hmat_re(:,:,:,4)*3, hmat_re(:,:,:,2), 'absolute', 2e-6 );
assertElementsAlmostEqual( hmat_re(:,:,:,4)*2, hmat_re(:,:,:,3), 'absolute', 1.5e-6 );

try 
    [~,~,~] = quadriga_lib.baseband_freq_response( real(coeff), imag(coeff), delay, pilots, fc, [2,3,2,1] );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Incorrect number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try 
    [~,~] = quadriga_lib.baseband_freq_response( real(coeff), imag(coeff), delay, pilots, fc, [2,3,2,1],1 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Incorrect number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try 
    [~,~] = quadriga_lib.baseband_freq_response( real(coeff), imag(coeff), single(delay), pilots, fc, [2,3,2,1] );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'All floating-point inputs must have the same type: ''single'' or ''double'' precision';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

[ hmat_re_s, hmat_im_s ] = quadriga_lib.baseband_freq_response( single(real(coeff)), single(imag(coeff)), single(delay), single(pilots), fc, [2,3,2,1] );

assertTrue( isa(hmat_re_s,'single') );
assertTrue( isa(hmat_im_s,'single') );

assertElementsAlmostEqual( hmat_re, double(hmat_re_s), 'absolute', 2e-5 );

try 
    [~,~] = quadriga_lib.baseband_freq_response( real(coeff), imag(coeff), delay(:), pilots, fc, [2,3,2,1] );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Coefficients and delays must have the same number of snapshots';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end



end

