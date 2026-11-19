function test_channel_export_obj_file

fn = 'test.obj';
if exist( fn,'file' ), delete(fn); end
if exist( 'test.mtl','file' ), delete('test.mtl'); end

chan = struct( 'rx_pos', rand(3,1), 'tx_pos', rand(3,1), 'coeff_re', rand(3,2,5,3), 'coeff_im', rand(3,2,5,3), ...
    'delay', zeros(3,2,5,3), 'n_interact', ones(5,3), 'interact_coord', rand(3,5,3), 'center_freq', 3.0e9 );

quadriga_lib.channel_export_obj_file( fn, chan );
quadriga_lib.channel_export_obj_file( fn, chan, [], [], [], [], [], [], [], [] );
assertEqual( fileread(fn), obj_text( chan ) ); % [] gives the defaults
quadriga_lib.channel_export_obj_file( fn, chan, 0 );
quadriga_lib.channel_export_obj_file( fn, chan, 1 );
quadriga_lib.channel_export_obj_file( fn, chan, [], 10, 6 );

% Zero-padded columns of interact_coord (as in the nD layout of snapshots with different interaction counts)
c = chan;
c.interact_coord(:,6:8,:) = 0;
quadriga_lib.channel_export_obj_file( fn, c );
assertEqual( fileread(fn), obj_text( chan ) );

% A channel from hdf5_read_channel can be exported directly
fn_h5 = 'test_channel_export_obj_file.hdf5';
if exist( fn_h5,'file' ), delete(fn_h5); end
quadriga_lib.hdf5_write_channel( fn_h5, chan );
quadriga_lib.channel_export_obj_file( fn, quadriga_lib.hdf5_read_channel( fn_h5 ) );
delete(fn_h5);

% The function returns no output; an output request is an error and writes no file
if exist( fn,'file' ), delete(fn); end
if exist( 'test.mtl','file' ), delete('test.mtl'); end
assert_cpp_error( @() call_with_output( fn, chan ), 'Wrong number of output arguments.' );
assertFalse( exist(fn, 'file') == 2 );

% Number of inputs
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn ), 'Wrong number of input arguments.' );
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, chan, [], [], [], [], [], [], [], [], [] ), ...
    'Wrong number of input arguments.' );

% The channel must be a 1x1 struct; C++ validates the rest (is_valid is tested in Catch2)
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, [chan, chan] ), 'Input ''chan'' must be a 1x1 struct.' );
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, 1 ), 'Input ''chan'' must be a 1x1 struct.' );
c = chan; c.delay = [];
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, c ), 'Delays are missing or incomplete.' ); % C++ check, passed through

% Options
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( '', chan ), 'OBJ-File name must end with .obj' );
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, chan, [], [], [], 'bls' ), 'Colormap is not supported.' );
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, chan, [], [], [], 'parula', [], [], [], 1 ), ...
    'Number of edges mut be >= 3.' );
quadriga_lib.channel_export_obj_file( fn, chan, [], [], [], [], [], 0 ); % paths without a volume
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, chan, [], [], [], [], [], -1 ), 'Radius cannot be negative.' );
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, chan, -1 ), 'Input ''max_paths'' must be a non-negative integer.' );

% Snapshot indices: 1-based in MATLAB, 0-based in C++
quadriga_lib.channel_export_obj_file( fn, chan, [], 10, 6, [], [1,1] );
assertEqual( numel( strfind( fileread(fn), '# Snapshot 0' ) ), 2 );
quadriga_lib.channel_export_obj_file( fn, chan, [], [], [], [], 3 );
txt = fileread(fn);
assertEqual( numel( strfind( txt, '# Snapshot' ) ), 1 );
assertEqual( numel( strfind( txt, '# Snapshot 2' ) ), 1 );
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, chan, [], 10, 6, [], 4 ), ...
    'Snapshot indices ''i_snap'' cannot exceed the number of snapshots in the channel.' );
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, chan, [], 10, 6, [], 0 ), ...
    'Entries in ''i_snap'' cannot be 0 (1-based index).' );
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, chan, [], 10, 6, [], [1 0] ), ...
    'Entries in ''i_snap'' cannot be 0 (1-based index).' );
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, chan, [], 10, 6, [], 1.5 ), ...
    'Entries in ''i_snap'' must be positive integers.' );

% Malformed fields are rejected by the MEX helpers
c = chan; c.n_interact = -c.n_interact;
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, c ), 'must be integers from 0 to 4294967295.' );
c = chan; c.n_interact = sparse(c.n_interact);
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, c ), 'is sparse' );
c = chan; c.coeff_re = ones(3,2,5,1,3);
assert_cpp_error( @() quadriga_lib.channel_export_obj_file( fn, c ), 'has more than 4 dimensions' );

if exist( fn,'file' ), delete(fn); end
if exist( 'test.mtl','file' ), delete('test.mtl'); end

end

% Text of the OBJ file written for chan under another name, with the material file name of 'test.obj'
function txt = obj_text( chan, varargin )
quadriga_lib.channel_export_obj_file( 'test_ref.obj', chan, varargin{:} );
txt = strrep( fileread('test_ref.obj'), 'test_ref.mtl', 'test.mtl' );
delete('test_ref.obj'); delete('test_ref.mtl');
end

function call_with_output( fn, chan )
x = quadriga_lib.channel_export_obj_file( fn, chan ); %#ok<NASGU>
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
