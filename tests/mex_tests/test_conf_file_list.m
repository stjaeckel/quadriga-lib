function test_conf_file_list
% MEX interface test for quadriga_lib.conf_file_list.
% Scope: argument and output handling, string and cell inputs, defaults, error paths.
% The listing rules are validated in the Catch2 suite (test_conf_file_list.cpp).

%% Setup: two directories; X.conf exists in both, b/X.conf is shadowed
da = 'test_conf_file_list_mex_a';
db = 'test_conf_file_list_mex_b';
cleanup( da, db );
mkdir( da );
mkdir( db );
write_file( [da, '/X.conf'], { 'ShortName = Xs' } );
write_file( [da, '/b.conf'], { 'DS_mu = -7' } );
write_file( [db, '/X.conf'], { 'ShortName = Other' } );
write_file( [db, '/A.conf'], { 'ShortName = As', 'ShortName = At' } );
write_file( [db, '/notes.txt'], { 'ShortName = No' } );

%% Basic call, all outputs
[ scenarios, file_names, file_dirs ] = quadriga_lib.conf_file_list( { da, db } );
assertEqual( scenarios, {'X'; 'Xs'; 'b'; 'A'; 'As'; 'At'} );
assertEqual( file_names, {'X.conf'; 'X.conf'; 'b.conf'; 'A.conf'; 'A.conf'; 'A.conf'} );
assertEqual( file_dirs, {da; da; da; db; db; db} );
assertTrue( iscell(scenarios) && isequal(size(scenarios), [6, 1]) );

% The order of the directories matters
scenarios2 = quadriga_lib.conf_file_list( { db, da } );
assertEqual( scenarios2, {'A'; 'As'; 'At'; 'X'; 'Other'; 'b'} );

%% Input 'dirs': one string, row and column cell, empty cell
[ scenarios3, ~, file_dirs3 ] = quadriga_lib.conf_file_list( da );
assertEqual( scenarios3, {'X'; 'Xs'; 'b'} );
assertEqual( file_dirs3, {da; da; da} );
assertEqual( quadriga_lib.conf_file_list( { da; db } ), scenarios );
[ scenarios4, file_names4, file_dirs4 ] = quadriga_lib.conf_file_list( {} );
assertTrue( iscell(scenarios4) && isempty(scenarios4) );
assertTrue( iscell(file_names4) && isempty(file_names4) );
assertTrue( iscell(file_dirs4) && isempty(file_dirs4) );
assertTrue( isempty( quadriga_lib.conf_file_list( 'test_conf_file_list_mex_missing' ) ) );

% An empty string is the current directory, as a string and inside a cell
fn = 'test_conf_file_list_mex_cwd.conf';
write_file( fn, { 'DS_mu = -7' } );
[ scenarios5, ~, file_dirs5 ] = quadriga_lib.conf_file_list( '' );
i5 = find( strcmp( scenarios5, 'test_conf_file_list_mex_cwd' ) );
[ scenarios6, ~, file_dirs6 ] = quadriga_lib.conf_file_list( { da, '' } );
i6 = find( strcmp( scenarios6, 'test_conf_file_list_mex_cwd' ) );
delete( fn );
assertEqual( numel(i5), 1 );
assertEqual( file_dirs5{i5}, '' );
assertEqual( numel(i6), 1 );
assertEqual( file_dirs6{i6}, '' );
assertEqual( scenarios6(1:3), {'X'; 'Xs'; 'b'} );

%% Optional input 'parse_shortnames': omitted, empty, explicit
assertEqual( quadriga_lib.conf_file_list( { da, db }, [] ), scenarios );
assertEqual( quadriga_lib.conf_file_list( { da, db }, true ), scenarios );
[ scenarios7, file_names7, file_dirs7 ] = quadriga_lib.conf_file_list( { da, db }, false );
assertEqual( scenarios7, {'X'; 'b'; 'A'} );
assertEqual( file_names7, {'X.conf'; 'b.conf'; 'A.conf'} );
assertEqual( file_dirs7, {da; da; db} );

%% Outputs: none, and fewer than the maximum
quadriga_lib.conf_file_list( { da, db } );
[ ~, file_names8 ] = quadriga_lib.conf_file_list( { da, db } );
assertEqual( file_names8, file_names );

%% Type handling
assertEqual( quadriga_lib.conf_file_list( { da, db }, 0 ), scenarios7 );
assertEqual( quadriga_lib.conf_file_list( { da, db }, uint32(1) ), scenarios );

%% Errors
assert_throws( @() quadriga_lib.conf_file_list(), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.conf_file_list( da, true, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out4( da ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.conf_file_list( 5 ), 'Input ''dirs'' must be a string or a cell array of strings.' );
assert_throws( @() quadriga_lib.conf_file_list( [] ), 'Input ''dirs'' must be a string or a cell array of strings.' );
assert_throws( @() quadriga_lib.conf_file_list( { da, 5 } ), 'must be a string' );
write_file( [db, '/C.conf'], { 'ShortName = Xs' } );                        % short name of a/X.conf, thrown by C++
assert_throws( @() quadriga_lib.conf_file_list( { da, db } ), 'ambiguous' );
assertEqual( quadriga_lib.conf_file_list( { da, db }, false ), {'X'; 'b'; 'A'; 'C'} );

cleanup( da, db );

end

% Write the lines of a cell array to a text file
function write_file(fn, lines)
fid = fopen( fn, 'w' );
for n = 1:numel(lines)
    fprintf( fid, '%s\n', lines{n} );
end
fclose( fid );
end

% Remove the test directories with their content
function cleanup(varargin)
for n = 1:numel(varargin)
    if exist( varargin{n}, 'dir' )
        f = dir( varargin{n} );
        for m = 1:numel(f)
            if ~f(m).isdir
                delete( [varargin{n}, '/', f(m).name] );
            end
        end
        rmdir( varargin{n} );
    end
end
end

% Request 4 outputs so that the nlhs > 3 guard fires
function out4(varargin)
[~, ~, ~, ~] = quadriga_lib.conf_file_list( varargin{:} ); %#ok<ASGLU>
end

% Assert that fn() raises an error whose message contains expected (substring).
function assert_throws(fn, expected)
try
    fn();
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expected)) %#ok<STREMP>
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expected, '", GOT: "', ME.message, '"']);
    end
end
end
