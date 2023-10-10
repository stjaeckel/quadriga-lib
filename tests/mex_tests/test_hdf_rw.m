function test_hdf_rw

verbose = 0;
run_tests = inf;

% MOxUnit Setup
tmp = which('MOxUnitTestSuite');
if isempty(tmp)
    current_dir = pwd;
    cd('../external/MOxUnit-master/MOxUnit');
    moxunit_set_path();
    cd(current_dir);
end

fn = 'hdf_mex.hdf5';
warning('off','quadriga_lib:hdf5_write_channel:overwriting_exisiting_data');

% Delete file
if exist( fn,'file' )
    delete(fn);
end

% Try creaating the file
tst = 0; if tst > run_tests; return; end
if verbose; disp("Test: Create file"); end
quadriga_lib.hdf5_create_file(fn);

% Read layout
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Create file"); end
storage_space = quadriga_lib.hdf5_read_layout(fn);
assertEqual(storage_space,uint32([65536, 1, 1, 1])); % Default

% Trying this again should fail because file exists
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Create file again"); end
try
    quadriga_lib.hdf5_create_file(fn);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Try creating a file with a custom storage layout
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Create file custom layout"); end
delete(fn);
quadriga_lib.hdf5_create_file(fn,[12,12]);
storage_space = quadriga_lib.hdf5_read_layout(fn);
assertEqual(storage_space,uint32([12, 12, 1, 1]));

% Reshape the storage layout
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Reshape layout"); end
[~] = quadriga_lib.hdf5_reshape_layout(fn,[1,1,18,8]);
storage_space = quadriga_lib.hdf5_read_layout(fn);
assertEqual(storage_space,uint32([1, 1, 18, 8]));

% There shoudl be an error if number of elements dont match
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Reshape layout with wrong size"); end
try
    quadriga_lib.hdf5_reshape_layout(fn,145);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

delete(fn);

% Calling the reshape function on a non-exisitng file should cause error
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Reshape layout in non-existing file"); end
try
    quadriga_lib.hdf5_reshape_layout(fn,144);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Test writing unstructured data
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Writing unstructured data"); end
par = struct;
par.string = 'Buy Bitcoin!';
par.double = 21e6;
par.single = single(pi);
par.uint32 = uint32(21);
par.int32 = int32(-11001001);
par.uint64 = uint64( 21e6*100e6 );
par.int64 = -int64( 21e6*100e6 );
par.double_Col = [0:0.1:1]';
par.single_Col = -single([0:0.1:1]');
par.uint32_Col = uint32([14:18]');
par.int32_Col = -int32([14:18]');
par.uint64_Col = uint64( 21e6*100e6 + [0,1]' );
par.int64_Col = -int64( 21e6*100e6 + [0,1]' );
par.double_Row = [1:0.1:2];
par.single_Row = -single([1:0.1:2]);
par.uint32_Row = uint32([17:19]);
par.int32_Row = -int32([12:19]);
par.uint64_Row = uint64( 21e6*100e6 + [2,3] );
par.int64_Row = -int64( 21e6*100e6 + [3,4] );
par.double_Mat = rand(4);
par.single_Mat = -single(rand(5));
par.uint32_Mat = randi(10,3,'uint32');
par.int32_Mat = -randi(10,4,'int32');
par.uint64_Mat = uint64(randi(10,5,'uint32')) + uint64( 21e6*100e6) ;
par.int64_Mat = int64(randi(10,6,'int32'))  - int64( 21e6*100e6) ;
par.double_Cube = rand(4,3,2);
par.single_Cube = -single(rand(5,4,3));
par.uint32_Cube = randi(10,3,3,4,'uint32');
par.int32_Cube = -randi(10,4,5,6,'int32');
par.uint64_Cube = uint64(randi(10,5,6,7,'uint32')) + uint64( 21e6*100e6) ;
par.int64_Cube = int64(randi(10,6,7,8,'int32'))  - int64( 21e6*100e6) ;

storage_space = quadriga_lib.hdf5_write_channel(fn,[1,1,1,1],par);
assertEqual( storage_space, uint32([128 8 8 8]) );

% More than 1 output shoud cause error
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Writing unstructured data with 2 outputs"); end
try
    [storage_space, ~] = quadriga_lib.hdf5_write_channel(fn,[2,1,1,1],par);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Try writing empty par - should be OK
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Writing empty channel"); end
quadriga_lib.hdf5_write_channel(fn,[2,1,1,1],[]);
assertEqual( storage_space, uint32([128 8 8 8]) );

% Read the data again and compare the results
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Reading channel"); end
parR = quadriga_lib.hdf5_read_channel(fn);

fieldsPar = fieldnames(par);
fieldsParR = fieldnames(parR);

% Read the names of the par
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Reading dataset names"); end
par_names = quadriga_lib.hdf5_read_dset_names(fn);

assertEqual( length(fieldsPar), length(fieldsParR) );       % Same number of fields
assertEqual( length(fieldsPar), length(par_names) );        % Same number of fields

assertTrue( all(strcmp(fieldsPar, fieldsParR)) );           % Same field names
assertTrue( all(strcmp(fieldsPar, par_names)) );            % Same field names

tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Reading datasets and writing datasets"); end
for n = 1:length(fieldsPar)
    field = fieldsPar{n};
    assertEqual( class(par.(field)), class(parR.(field)));  % Same data type
    assertTrue(  isequal(par.(field), parR.(field)) );      % Same data

    % Load single fields
    data = quadriga_lib.hdf5_read_dset(fn, 1, field);
    assertTrue(  isequal(par.(field), data) );      % Same data

    % Add a copy of the data to new storage location
    quadriga_lib.hdf5_write_dset(fn, [1,2], field, data);
end

% Check if number of datasets matches
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Reading dataset names from 2nd location"); end
par_names = quadriga_lib.hdf5_read_dset_names(fn,[1,2]);
assertTrue( numel(par_names) == 31 );

% Overwriting an exisiting dataset should cause error
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Overwriting existing dataset"); end
try
    quadriga_lib.hdf5_write_dset(fn, [1,2], 'string', 'Oh no, I bought Ethereum.');
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Reading from a empty location
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Reading dataset from empty location"); end
par_names = quadriga_lib.hdf5_read_dset_names(fn,[12,2,2]);
assertTrue(isempty(par_names));

% Passing a snapshot range should work fine since there is no structured data
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Snapshot range for unstructured data"); end
quadriga_lib.hdf5_read_channel(fn,1,2);

% Trying to write a complex number should greate an error
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Complex data should fail"); end
parC.complex = 1 + 21i;
try
    quadriga_lib.hdf5_write_channel(fn,[2,1,1,1],parC);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

rx_pos = rand(3,1);
tx_pos = rand(3,1);

% Only providing a rx_pos should lead to an error
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Only rx location should fail"); end
try
    quadriga_lib.hdf5_write_channel(fn,[2,1,1,1],[],rx_pos);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% This should be OK, but useless
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Tx and rx location"); end
quadriga_lib.hdf5_write_channel(fn,[2,1,1,1],[],rx_pos,tx_pos);

% % Rewriting the data to the same position should issue a warning (does not work for octave)
% lastwarn(''); % Clear last warning
% quadriga_lib.hdf5_write_channel(fn,[2,1,1,1],[],rx_pos,tx_pos);
% [warnMsg, ~] = lastwarn;
% if isempty(warnMsg)
%     error('moxunit:exceptionNotRaised', 'Expected a warning!');
% end

coeff_re = rand(3,2,5,3);
coeff_im = rand(3,2,5,3);

% Pssing only coeff_re should cause error
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Only real-valued coefficients"); end
try
    quadriga_lib.hdf5_write_channel(fn,[3,1,1,1],[],rx_pos,tx_pos,coeff_re);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Pssing only coeff_im should cause error
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Missing real-valued coefficients"); end
try
    quadriga_lib.hdf5_write_channel(fn,[3,1,1,1],[],rx_pos,tx_pos,[],coeff_im);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Pssing only coeff_re and coeff_im without delays should cause error
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Missing delays"); end
try
    quadriga_lib.hdf5_write_channel(fn,[3,1,1,1],[],rx_pos,tx_pos,coeff_re,coeff_im);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% This should work fine
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Compledte structured set"); end
delay_4d = rand(3,2,5,3);
quadriga_lib.hdf5_write_channel(fn, [3,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d);

% Test if we can restore the data
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Restore data"); end
[~, rx_posR, tx_posR, coeff_reR, coeff_imR, delay_4dR ] = quadriga_lib.hdf5_read_channel(fn, 3);

assertEqual( rx_posR, single(rx_pos) );
assertEqual( tx_posR, single(tx_pos) );
assertEqual( coeff_reR, single(coeff_re) );
assertEqual( coeff_imR, single(coeff_im) );
assertEqual( delay_4dR, single(delay_4d) );

% Test if we can restore the data in reverse order
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Restore data in reverse order"); end
[~, rx_posR, tx_posR, coeff_reR, coeff_imR, delay_4dR ] = quadriga_lib.hdf5_read_channel(fn, 3, [3,2,1]);

assertEqual( rx_posR, single(rx_pos) );
assertEqual( tx_posR, single(tx_pos) );
assertEqual( coeff_reR, single(coeff_re(:,:,:,[3,2,1])) );
assertEqual( coeff_imR, single(coeff_im(:,:,:,[3,2,1])) );
assertEqual( delay_4dR, single(delay_4d(:,:,:,[3,2,1])) );

% Test out-of-bound error
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Out of bound"); end
try
    quadriga_lib.hdf5_read_channel(fn, 3, 0);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_read_channel(fn, 3, 4);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Test alternative delays
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: 2D Delays"); end
delay_2d = rand(5,3);
quadriga_lib.hdf5_write_channel(fn, [4,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_2d);
[~, ~, ~, ~, ~, delay_2dR ] = quadriga_lib.hdf5_read_channel(fn, 4);
assertEqual( squeeze(delay_2dR), single(delay_2d) );

% Add optional parameters
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Writing optional parameters"); end
center_frequency = 21e9;
name = 'buy_more_bitcoin';
path_gain = rand(5,3);
path_length = rand(5,3);
path_polarization = rand(8,5,3);
path_angles = rand(5,4,3);
path_coord = rand(3,12,5,3);
rx_orientation = rand(3,3);
tx_orientation = rand(3,3);

quadriga_lib.hdf5_write_channel(fn, [5,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d, ...
    center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord, rx_orientation, tx_orientation );

% Test if we can restore the data
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Reading optional parameters"); end
[parR, rx_posR, tx_posR, coeff_reR, coeff_imR, delay_4dR, center_frequencyR, nameR, initR, path_gainR, path_lengthR, ...
    path_polarizationR, path_anglesR, path_coordR, rx_orientationR, tx_orientationR ] =...
    quadriga_lib.hdf5_read_channel(fn, 5);

assertTrue( isempty(parR) );
assertEqual( rx_posR, single(rx_pos) );
assertEqual( tx_posR, single(tx_pos) );
assertEqual( coeff_reR, single(coeff_re) );
assertEqual( coeff_imR, single(coeff_im) );
assertEqual( delay_4dR, single(delay_4d) );
assertEqual( center_frequencyR, single(center_frequency) );
assertEqual( nameR, name );
assertEqual( initR, int32(0) );
assertEqual( path_gainR, single(path_gain) );
assertEqual( path_lengthR, single(path_length) );
assertEqual( path_polarizationR, single(path_polarization) );
assertEqual( path_anglesR, single(path_angles) );
assertEqual( path_coordR, single(path_coord) );
assertEqual( rx_orientationR, single(rx_orientation) );
assertEqual( tx_orientationR, single(tx_orientation) );

% Test if we can restore a single snapsot
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Load single snapshot"); end
[~, rx_posR, tx_posR, coeff_reR, coeff_imR, delay_4dR, center_frequencyR, nameR, initR, path_gainR, path_lengthR, ...
    path_polarizationR, path_anglesR, path_coordR, rx_orientationR, tx_orientationR ] =...
    quadriga_lib.hdf5_read_channel(fn, 5, 3);

assertEqual( rx_posR, single(rx_pos) );
assertEqual( tx_posR, single(tx_pos) );
assertEqual( coeff_reR, single(coeff_re(:,:,:,3)) );
assertEqual( coeff_imR, single(coeff_im(:,:,:,3)) );
assertEqual( delay_4dR, single(delay_4d(:,:,:,3)) );
assertEqual( center_frequencyR, single(center_frequency) );
assertEqual( nameR, name );
assertEqual( initR, int32(0) );
assertEqual( path_gainR, single(path_gain(:,3)) );
assertEqual( path_lengthR, single(path_length(:,3)) );
assertEqual( path_polarizationR, single(path_polarization(:,:,3)) );
assertEqual( path_anglesR, single(path_angles(:,:,3)) );
assertEqual( path_coordR, single(path_coord(:,:,:,3)) );
assertEqual( rx_orientationR, single(rx_orientation(:,3)) );
assertEqual( tx_orientationR, single(tx_orientation(:,3)) );

% Test ill-formatted coeff_re
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Ill-formatted coeff-re"); end
try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re([1,2],:,:,:), coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re(:,[1,1,1],:,:), coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re(:,:,[1,2,3,4],:), coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re(:,:,:,[1,2]), coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Test ill-formatted coeff_im
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Ill-formatted coeff-im"); end
try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im([1,2],:,:,:), delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im(:,[1,1,1],:,:), delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im(:,:,[1,2,3,4],:), delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im(:,:,:,[1,2]), delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Test ill-formatted delay_4d
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Ill-formatted 4D delays"); end
try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d([1,2],:,:,:), ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d(:,[1,1,1],:,:), ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d(:,:,[1,2,3,4],:), ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d(:,:,:,[1,2]), ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Test ill-formatted delay_2d
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Ill-formatted 2D delays"); end
try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_2d([1,2],:), ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_2d(:,[1,2]), ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Test ill-formatted path gain
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Ill-formatted path gain"); end
try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain(:,[1,2]), path_length, path_polarization, path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain([1,2,3,4],:), path_length, path_polarization, path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Test ill-formatted path length
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Ill-formatted path lenght"); end
try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain, path_length(:,[1,2]), path_polarization, path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain, path_length([1,2,3,4],:), path_polarization, path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Test ill-formatted path polarization
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Ill-formatted polarizaiion"); end
try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization([1,2,3],:,:), path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization(:,[1,2,3,4],:), path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization(:,:,[1,2]), path_angles, path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Test ill-formatted path angles
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Ill-formatted angles"); end
try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles([1,2,3],:,:), path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles(:,[1,2,3],:), path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles(:,:,[1,2]), path_coord, rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Test ill-formatted path_coord
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Ill-formatted coords"); end
try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord([1,2],:,:,:), rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord(:,:,[1,2,3],:), rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord(:,:,:,[1,2]), rx_orientation, tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Test ill-formatted rx_orientation
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Ill-formatted rx orientation"); end
try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord, rx_orientation([1,2],:), tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord, rx_orientation(:,[1,2]), tx_orientation );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Test ill-formatted tx_orientation
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Ill-formatted tx orientation"); end
try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord, rx_orientation, tx_orientation([1,2],:) );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.hdf5_write_channel(fn, [6,1,1,1], [], rx_pos, tx_pos, coeff_re, coeff_im, delay_4d, ...
        center_frequency, name, [], path_gain, path_length, path_polarization, path_angles, path_coord, rx_orientation, tx_orientation(:,[1,2]) );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Test alternative format
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Alternative format"); end
rx_pos = rand(3,3);
tx_pos = rand(3,3);
center_frequency = rand(3,1);

quadriga_lib.hdf5_write_channel(fn, [6,1,1,2], par, rx_pos, tx_pos, coeff_re, coeff_im, delay_4d, ...
    center_frequency, name, 2, [], [], [], [], [], rx_orientation(:,1), tx_orientation(:,2) );

% Test if we can restore the data
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Alternative format loading"); end
[parR, rx_posR, tx_posR, coeff_reR, coeff_imR, delay_4dR, center_frequencyR, nameR, initR, path_gainR, path_lengthR, ...
    path_polarizationR, path_anglesR, path_coordR, rx_orientationR, tx_orientationR ] =...
    quadriga_lib.hdf5_read_channel(fn, [6,1,1,2]);

assertTrue( ~isempty(parR) );
assertEqual( rx_posR, single(rx_pos) );
assertEqual( tx_posR, single(tx_pos) );
assertEqual( coeff_reR, single(coeff_re) );
assertEqual( coeff_imR, single(coeff_im) );
assertEqual( delay_4dR, single(delay_4d) );
assertEqual( center_frequencyR, single(center_frequency) );
assertEqual( nameR, name );
assertEqual( initR, int32(2) );
assertTrue( isempty(path_gainR) );
assertTrue( isempty(path_lengthR) );
assertTrue( isempty(path_polarizationR) );
assertTrue( isempty(path_anglesR) );
assertTrue( isempty(path_coordR) );
assertEqual( rx_orientationR, single(rx_orientation(:,1)) );
assertEqual( tx_orientationR, single(tx_orientation(:,2)) );

% Read storage space
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Reading layout "); end
[storage_space, has_data] = quadriga_lib.hdf5_read_layout(fn);
assertEqual( storage_space, uint32([128 8 8 8]) );
assertTrue( all(has_data(1:5,1,1,1)) );
assertTrue( has_data(6,1,1,2) == uint32(1) );
assertTrue( has_data(1,2,1,1) == uint32(1) );
assertTrue( sum(has_data(:)) == uint32(7) );

% This shoudl return empty stuff
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Reading empty layout"); end
[storage_space, has_data] = quadriga_lib.hdf5_read_layout('bla.hdf5');
assertEqual( storage_space, uint32([0 0 0 0]) );
assertTrue( isempty(has_data) );

% Loading a non-HDF5 files should cause an error
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: reading non-hdf5 file"); end
delete(fn);
files = dir;
try
    quadriga_lib.hdf5_read_layout(files(end).name);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

end