function test_cart2geo

e = rand(3,6,2);

f = @() quadriga_lib.cart2geo;
assertExceptionThrown( f, 'quadriga_tools:cart2geo:no_input')

f = @() quadriga_lib.cart2geo(e,[]);
assertExceptionThrown( f, 'quadriga_tools:cart2geo:no_input')

f = @() quadriga_lib.cart2geo(e);
assertExceptionThrown( f, 'quadriga_tools:cart2geo:no_output')

% empty
try
    [~,~] = quadriga_lib.cart2geo([]);
    error_exception_not_thrown('quadriga_tools:cart2geo:empty');
catch expt
    error_if_wrong_id_thrown('quadriga_tools:cart2geo:empty',expt.identifier);
end

% size_mismatch
try
    [~,~] = quadriga_lib.cart2geo(rand(2,2,2,2));
    error_exception_not_thrown('quadriga_tools:cart2geo:size_mismatch');
catch expt
    error_if_wrong_id_thrown('quadriga_tools:cart2geo:size_mismatch',expt.identifier);
end
try
    [~,~] = quadriga_lib.cart2geo(rand(2,2,2));
    error_exception_not_thrown('quadriga_tools:cart2geo:size_mismatch');
catch expt
    error_if_wrong_id_thrown('quadriga_tools:cart2geo:size_mismatch',expt.identifier);
end

% wrong_type
try
    [~,~] = quadriga_lib.cart2geo(int32([1;1;1]));
    error_exception_not_thrown('quadriga_tools:cart2geo:wrong_type');
catch expt
    error_if_wrong_id_thrown('quadriga_tools:cart2geo:wrong_type',expt.identifier);
end

[az,el] = quadriga_lib.cart2geo(e);
assert( isa(az,'double') );
assert( isa(el,'double') );

[az,el,le] = quadriga_lib.cart2geo(single(e));
assert( isa(az,'single') );
assert( isa(el,'single') );
assert( isa(le,'single') );

v = 1/sqrt(2);
[az,el,le] = quadriga_lib.cart2geo([-v;-v;0]);
assertElementsAlmostEqual( az, -3*pi/4, 'absolute', 1e-5 );
assertElementsAlmostEqual( el, 0, 'absolute', 1e-5 );
assertElementsAlmostEqual( le, 1, 'absolute', 1e-5 );

[az,el,le] = quadriga_lib.cart2geo([0;2*v;2*v]);
assertElementsAlmostEqual( az, pi/2, 'absolute', 1e-5 );
assertElementsAlmostEqual( el, pi/4, 'absolute', 1e-5 );
assertElementsAlmostEqual( le, 2, 'absolute', 1e-5 );

[az,el,le] = quadriga_lib.cart2geo([0,v;1,v;1,-1]);
assertElementsAlmostEqual( az, [pi/2;pi/4], 'absolute', 1e-5 );
assertElementsAlmostEqual( el, pi/4*[1;-1], 'absolute', 1e-5 );
assertElementsAlmostEqual( le, sqrt(2)*[1;1], 'absolute', 1e-5 );


% ---------------- HELPER FUNCTIONS ------------------
function error_exception_not_thrown(error_id)
error('moxunit:exceptionNotRaised', 'Exception ''%s'' not thrown', error_id);

function error_if_wrong_id_thrown(expected_error_id, thrown_error_id)
if ~strcmp(thrown_error_id, expected_error_id)
    error('moxunit:wrongExceptionRaised',...
        'Exception raised with id ''%s'' expected id ''%s''',...
        thrown_error_id,expected_error_id);
end
