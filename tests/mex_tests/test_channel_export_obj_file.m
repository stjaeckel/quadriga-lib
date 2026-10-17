function test_channel_export_obj_file

fn = 'test.obj';
if exist( fn,'file' )
    delete(fn);
end
if exist( 'test.mtl','file' )
    delete('test.mtl');
end

rx_pos = rand(3,1);
tx_pos = rand(3,1);
coeff_re = rand(3,2,5,3);
coeff_im = rand(3,2,5,3);
no_interact = ones(5,3);
interact_coord = rand(3,5,3);
center_freq = 3.0e9;

quadriga_lib.channel_export_obj_file( fn, [], [], [], [] ,[], [], [], ...
    rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );

quadriga_lib.channel_export_obj_file( fn, 0, [], [], [] ,[], [], [], ...
    rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );

quadriga_lib.channel_export_obj_file( fn, 1, [], [], [] ,[], [], [], ...
    rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );

quadriga_lib.channel_export_obj_file( fn, [], 10, 6, [] ,[], [], [], ...
    rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );

% Test ill-formatted file name
try
    quadriga_lib.channel_export_obj_file( '', [], [], [], [] ,[], [], [], ...
        rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'OBJ-File name must end with .obj';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Test wrong colormap
try
    quadriga_lib.channel_export_obj_file( fn, [], [], [], 'bls' ,[], [], [], ...
        rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Colormap is not supported.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Test wrong number of edges
try
    quadriga_lib.channel_export_obj_file( fn, [], [], [], 'parula' ,[], [], 1, ...
        rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of edges mut be >= 3.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% This should create objects that only contain paths without a volume
quadriga_lib.channel_export_obj_file( fn, [], [], [], [] , 0, [], [], ...
    rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );

% Test negative radius
try
    quadriga_lib.channel_export_obj_file( fn, [], [], [], [] , -1, [], [], ...
        rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Radius cannot be negative.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Test ill-formatted rx position
try
    quadriga_lib.channel_export_obj_file( fn, [], [], [], [] , [], [], [], ...
        rand(2,1), tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = '''rx_pos'' is missing or ill-formatted (must have 3 rows).';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Test ill-formatted interact_coord
try
    quadriga_lib.channel_export_obj_file( fn, [], [], [], [], [], [], [], ...
        rand(2,1), tx_pos, no_interact, interact_coord(:,:,[1,2]), center_freq, coeff_re, coeff_im  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of snapshots in interact_coord must match coefficients.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

quadriga_lib.channel_export_obj_file( fn, [], 10, 6, [] ,[], [], [], ...
    rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im, [1,1]  );

% Snapshot index out of range
try
    quadriga_lib.channel_export_obj_file( fn, [], 10, 6, [] ,[], [], [], ...
        rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im, 4  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Snapshot indices ''i_snap'' cannot exceed the number of snapshots in the channel.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

delete(fn);
delete('test.mtl');

% Snapshot index 0: i_snap is 1-based in MATLAB and 0-based in C++
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, [], 10, 6, [] ,[], [], [], ...
    rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im, 0 ), ...
    'Entries in ''i_snap'' cannot be 0 (1-based index).' );
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, [], 10, 6, [] ,[], [], [], ...
    rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im, [1 0] ), ...
    'Entries in ''i_snap'' cannot be 0 (1-based index).' );
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, [], 10, 6, [] ,[], [], [], ...
    rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im, 1.5 ), ...
    'Entries in ''i_snap'' must be integers from 0 to' );

% no_interact and interact_coord need one entry per snapshot of the coefficients (empty no_interact crashed)
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, [], [], [], [] ,[], [], [], ...
    rx_pos, tx_pos, [], interact_coord, center_freq, coeff_re, coeff_im ), ...
    'Number of snapshots in no_interact must match coefficients.' );
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, [], [], [], [] ,[], [], [], ...
    rx_pos, tx_pos, no_interact(:,1:2), interact_coord, center_freq, coeff_re, coeff_im ), ...
    'Number of snapshots in no_interact must match coefficients.' );
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, [], [], [], [] ,[], [], [], ...
    rx_pos, tx_pos, ones(5,4), rand(3,5,4), center_freq, coeff_re, coeff_im ), ...
    'Number of snapshots in no_interact must match coefficients.' );
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, [], [], [], [] ,[], [], [], ...
    rx_pos, tx_pos, no_interact, rand(3,5,4), center_freq, coeff_re, coeff_im ), ...
    'Number of snapshots in interact_coord must match coefficients.' );

% Malformed input is rejected by the MEX helpers
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, [], [], [], [] ,[], [], [], ...
    rx_pos, tx_pos, -no_interact, interact_coord, center_freq, coeff_re, coeff_im ), ...
    'must be integers from 0 to 4294967295.' );
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, [], [], [], [] ,[], [], [], ...
    rx_pos, tx_pos, sparse(no_interact), interact_coord, center_freq, coeff_re, coeff_im ), ...
    'is sparse' );
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, -1, [], [], [] ,[], [], [], ...
    rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im ), ...
    'Input ''max_no_paths'' must be an integer from 0 to' );
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, [], [], [], [] ,[], [], [], ...
    rx_pos, tx_pos, no_interact, interact_coord, center_freq, ones(3,2,5,1,3), coeff_re ), ...
    'has more than 4 dimensions' );

if exist( fn,'file' ), delete(fn); end
if exist( 'test.mtl','file' ), delete('test.mtl'); end

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
