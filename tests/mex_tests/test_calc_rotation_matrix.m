function test_calc_rotation_matrix

f = @() quadriga_lib.calc_rotation_matrix;
assertExceptionThrown( f, 'quadriga_lib:calc_rotation_matrix:no_input')

% 4 inputs should throw error
try
    [~] = quadriga_lib.calc_rotation_matrix(1,1,1,1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% 2 outputs should throw error
try
    [~,~] = quadriga_lib.calc_rotation_matrix([0;0;0]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% empty input
try
    [~] = quadriga_lib.calc_rotation_matrix([]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% wrong input
try
    [~] = quadriga_lib.calc_rotation_matrix(eye(2));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% int32 input
try
    [~] = quadriga_lib.calc_rotation_matrix(int32([0;0;0]));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% invert_y_axis
[~] = quadriga_lib.calc_rotation_matrix([0;0;0],0);
[~] = quadriga_lib.calc_rotation_matrix([0;0;0],2);
[~] = quadriga_lib.calc_rotation_matrix([0;0;0],false);
[~] = quadriga_lib.calc_rotation_matrix([0;0;0],true);

c = cos(pi/8);
s = sin(pi/8);

R = quadriga_lib.calc_rotation_matrix([0;0;0]);
assert( isa(R,'double') );
assertElementsAlmostEqual( R, [1,0,0, 0,1,0, 0,0,1]', 'absolute', 1e-5 );

R = quadriga_lib.calc_rotation_matrix(single([pi/8;0;0]));
assert( isa(R,'single') );
assertElementsAlmostEqual( R, [1,0,0,0,c,s,0,-s,c]', 'absolute', 1e-5 );

R = quadriga_lib.calc_rotation_matrix([0,pi/8,0;0,0,pi/8]');
assertElementsAlmostEqual( R(:,1), [c,0,-s,0,1,0,s,0,c]', 'absolute', 1e-5 );
assertElementsAlmostEqual( R(:,2), [c,s,0,-s,c,0,0,0,1]', 'absolute', 1e-5 );

R = quadriga_lib.calc_rotation_matrix([0;-pi/8;0],1);
assertElementsAlmostEqual( R, [c,0,-s,0,1,0,s,0,c]', 'absolute', 1e-5 );

R = quadriga_lib.calc_rotation_matrix([0;-pi/8;0],0,1);
assertElementsAlmostEqual( R, [c,0,-s,0,1,0,s,0,c]', 'absolute', 1e-5 );
