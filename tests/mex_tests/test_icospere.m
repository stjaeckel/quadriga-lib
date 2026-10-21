function test_icospere

quadriga_lib.icosphere;

center = quadriga_lib.icosphere;
assertEqual( size(center), [20,3] );

center2 = quadriga_lib.icosphere([],2);
assertElementsAlmostEqual( 2*center, center2, 'absolute', 1e-14 );

[center, length] = quadriga_lib.icosphere(1,3);
assertElementsAlmostEqual( sqrt(sum(center.^2,2)), length, 'absolute', 1e-14 );

[center, ~, vert] = quadriga_lib.icosphere(2);
assertElementsAlmostEqual( sum((center + vert(:,1:3)).^2,2), ones(80,1), 'absolute', 1e-14 );
assertElementsAlmostEqual( sum((center + vert(:,4:6)).^2,2), ones(80,1), 'absolute', 1e-14 );
assertElementsAlmostEqual( sum((center + vert(:,7:9)).^2,2), ones(80,1), 'absolute', 1e-14 );

[~, ~, ~, dir] = quadriga_lib.icosphere(2);
assertTrue( all(all(abs(dir(:,[1,3,5]) )<=pi)) );
assertTrue( all(all(abs(dir(:,[2,4,6]) )<=pi/2)) );

[~, ~, ~, dir] = quadriga_lib.icosphere(2, [], 1);

assertTrue( all(sum(dir(:,1:3).^2,2) - 1 < 1e-14));
assertTrue( all(sum(dir(:,4:6).^2,2) - 1 < 1e-14));
assertTrue( all(sum(dir(:,7:9).^2,2) - 1 < 1e-14));


try % 4 imputs
    [~,~,~,~,~] = quadriga_lib.icosphere;
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 4 imputs
    quadriga_lib.icosphere(1,2,3,4);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

%% Integer inputs must be integers in the range of the C++ type
assert_cpp_error( @() quadriga_lib.icosphere( -1 ), 'Input ''n_div'' must be an integer from 0 to 18446744073709551615.' );
assert_cpp_error( @() quadriga_lib.icosphere( NaN ), 'Input ''n_div'' must be an integer from 0 to' );
assert_cpp_error( @() quadriga_lib.icosphere( 2.5 ), 'Input ''n_div'' must be an integer from 0 to' );
assert_cpp_error( @() quadriga_lib.icosphere( Inf ), 'Input ''n_div'' must be an integer from 0 to' );
assert_cpp_error( @() quadriga_lib.icosphere( 1e30 ), 'Input ''n_div'' must be an integer from 0 to' );
assert_cpp_error( @() quadriga_lib.icosphere( int32(-1) ), 'Input ''n_div'' must be an integer from 0 to' );
assert_cpp_error( @() quadriga_lib.icosphere( sparse(2) ), 'Input ''n_div'' is sparse' );
center = quadriga_lib.icosphere( 2.0 );                                    % integer-valued double is fine
assertEqual( size(center), [80,3] );
center = quadriga_lib.icosphere( uint64(2) );
assertEqual( size(center), [80,3] );

% The radius must be finite and positive; n_div must not overflow the output size
assert_cpp_error( @() quadriga_lib.icosphere( 1, 0 ), 'Input ''radius'' must be a finite positive number.' );
assert_cpp_error( @() quadriga_lib.icosphere( 1, NaN ), 'Input ''radius'' must be a finite positive number.' );
assert_cpp_error( @() quadriga_lib.icosphere( 1, Inf ), 'Input ''radius'' must be a finite positive number.' );
assert_cpp_error( @() quadriga_lib.icosphere( 2^32 ), 'Input ''n_div'' is too large.' );
end

% Assert that fn() raises quadriga_lib:CPPerror with a message that contains expected (substring).
function assert_cpp_error(fn, expected)
try
    fn();
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if ~strcmp(ME.identifier, 'quadriga_lib:CPPerror') || isempty(strfind(ME.message, expected)) %#ok<STREMP>
        error('moxunit:exceptionNotRaised', ['EXPECTED: quadriga_lib:CPPerror "', expected, '", GOT: ', ...
            ME.identifier, ' "', ME.message, '"']);
    end
end
end
