% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_eval_antennas
% MEX interface test for quadriga_lib.eval_antennas.
% Scope: argument and output handling, defaults, the arrays in the output, error paths.
% The evaluation rules, the bands and the messages are tested in the Catch2 suite (test_set_antenna.cpp).

%% Setup: 3GPP panel of 2 x 2 x 1 x 2 elements for the Tx nodes (set without evaluation), a dipole for Rx node 1
fn_tx = 'test_eval_antennas_tx.csv';
fn_rx = 'test_eval_antennas_rx.csv';
cleanup = onCleanup( @() delete_files( { fn_tx, fn_rx } ) ); %#ok<NASGU>
write_text( fn_tx, 'name,x,y,z,antenna,panel\nA,0,0,25,3gpp,"2,2,1,2"\nB,500,0,25,3gpp,"2,2,1,2"\n' );
write_text( fn_rx, 'name,x,y,z,antenna\nU1,10,20,1.5,dipole\nU2,30,40,1.5,\n' );
L = quadriga_lib.layout_csv_read( fn_tx, fn_rx, [], false );
assertEqual( size( L.antenna ), [ 0, 1 ] );
assertTrue( isempty( L.center_freq ) );

%% Basic call: the arrays are generated at the given frequencies
Le = quadriga_lib.eval_antennas( L, [ 2e9, 3.5e9 ] );
assertEqual( Le.center_freq, [ 2e9; 3.5e9 ] );
assertEqual( size( Le.antenna ), [ 2, 1 ] );
assertEqual( size( Le.antenna{1} ), [ 1, 2 ] );                                 % spaced array: one per band
assertEqual( size( Le.antenna{2} ), [ 1, 1 ] );                                 % one dipole for all bands
assertEqual( size( Le.antenna{1}(1).element_pos ), [ 3, 8 ] );
lambda = 299792458 ./ [ 2e9, 3.5e9 ];                                           % the spacing is in wavelengths
d1 = Le.antenna{1}(1).element_pos(2, :);
d2 = Le.antenna{1}(2).element_pos(2, :);
assertElementsAlmostEqual( ( max( d1 ) - min( d1 ) ) / lambda(1), ( max( d2 ) - min( d2 ) ) / lambda(2), 'absolute', 1e-12 );
assertEqual( Le.tx.i_antenna, uint32( [ 1; 1 ] ) );
assertEqual( Le.rx.i_antenna, uint32( [ 2; 0 ] ) );
assertEqual( L.center_freq, zeros( 0, 0 ) );                                    % the input stays unchanged
assertEqual( size( L.antenna ), [ 0, 1 ] );

%% Optional inputs: [] and omitted keep the frequencies of the layout; a new list regenerates all lines
Le2 = quadriga_lib.eval_antennas( Le );
assertTrue( isequaln( Le2, Le ) );
Le3 = quadriga_lib.eval_antennas( Le, [] );
assertTrue( isequaln( Le3, Le ) );
Le4 = quadriga_lib.eval_antennas( Le, [ 1e9; 3.5e9 ] );
assertEqual( Le4.center_freq, [ 1e9; 3.5e9 ] );
assertEqual( Le4.antenna{1}(1).center_freq, 1e9 );

%% All members of the layout survive the conversion to C++ and back: nodes with every column, antennas, links
write_text( fn_tx, [ 'name,sector,x,y,z,heading,tilt,antenna,panel,seed,layout_seed,wrap,d_clutter,r_clutter,h_clutter,scenpar.DS_VS,scenario,note\n', ...
    'A,0,0,0,25,30,10,3gpp,"2,2,1,2",7,42,"1000,0;0,1000",5,0.5,3,0.2,UMa,x\n', ...
    'A,1,,,,150,,,,,,,,,,,,\n', ...
    'B,0,1000,0,25,90,0,omni,,,,,,,,,UMa,y\n' ] );
write_text( fn_rx, [ 'name,x,y,z,heading,speed,direction_az,direction_el,time,indoor,floor,d2d_in,relative,pairing\n', ...
    'U1,100,0,1.5,0,,,,0,0,,,1,\n', ...
    'U1,110,0,1.5,,,,,1,,,,,\n', ...
    'U2,900,0,1.5,90,2,45,10,,0,,,,A\n' ] );
Lr = quadriga_lib.layout_csv_read( fn_tx, fn_rx, [ 2e9, 3.5e9 ] );
Lr = quadriga_lib.init_link_states( Lr, '38.901', [], false, 2e9, '', 5 );
Lr.name = 'rich';
assertEqual( Lr.tx.seed, int64( [ 7; 7; -1 ] ) );
assertEqual( Lr.tx.clutter, [ 5, 5, NaN; 0.5, 0.5, NaN; 3, 3, NaN ] );
assertEqual( fieldnames( Lr.tx.scenpar_override )', { 'DS_VS' } );
assertEqual( Lr.tx.scenpar_override.DS_VS, [ 0.2; 0.2; NaN ] );
assertEqual( Lr.rx.relative, uint32( [ 1; 0 ] ) );
assertEqual( Lr.seed, int64( 5 ) );                                              % init_link_states stores the master seed
assertEqual( Lr.wrap, [ 1000, 0; 0, 1000 ] );
assertEqual( Lr.pairing, uint64( [ 1, 2, 1; 1, 1, 2 ] ) );                        % pairs (site, terminal), 1-based: U1 with A and B, U2 with A
assertEqual( Lr.rx.time, [ 0; 1; NaN ] );
assertEqual( Lr.tx.extra, { { 'x'; ''; 'y' } } );
assertFalse( isempty( Lr.links.i_tx ) || isempty( Lr.antenna ) );
Lr2 = quadriga_lib.eval_antennas( Lr );
assertTrue( isequaln( Lr2, Lr ) );

%% links = [] discards the links; missing and empty fields are empty members; unknown fields of nested structs are rejected
Ld = Lr;
Ld.links = [];
Ld = quadriga_lib.eval_antennas( Ld );
assertTrue( isempty( Ld.links.i_tx ) && isempty( Ld.links.method ) && isempty( Ld.links.conf_names ) );
assertEqual( Ld.links.seed, int64( -1 ) );
assertEqual( Ld.links.anchor_freq, 2e9 );
assertEqual( Ld.tx, Lr.tx );
Lm = quadriga_lib.eval_antennas( struct( 'center_freq', 2e9 ) );                % every other field is missing
assertEqual( Lm.center_freq, 2e9 );
assertEqual( Lm.tx.name, cell( 0, 1 ) );
assertTrue( isempty( Lm.rx.pos ) && isempty( Lm.tx.offset ) && isempty( Lm.pairing ) && isempty( Lm.antenna ) );
assertEqual( Lm.seed, int64( -1 ) );
Lz = Lr;
Lz.tx.clutter = [];
Lz.tx.scenpar_override = [];
Lz.rx.time = [];
Lz = quadriga_lib.eval_antennas( Lz );
assertTrue( isempty( Lz.tx.clutter ) && isempty( Lz.rx.time ) && isempty( fieldnames( Lz.tx.scenpar_override ) ) );
assertEqual( Lz.tx.pos, Lr.tx.pos );
for part = { 'tx', 'rx', 'links' }
    Lu = Lr;
    Lu.(part{1}).foo = 1;
    assert_throws( @() quadriga_lib.eval_antennas( Lu ), [ 'Unknown field ''foo'' in ''layout_in.', part{1}, '''.' ] );
end

%% Outputs: none
quadriga_lib.eval_antennas( L, [ 2e9, 3.5e9 ] );
assertTrue( isequaln( ans, Le ) );

%% Type handling: single frequencies
Ls = quadriga_lib.eval_antennas( L, single( [ 2e9; 3.5e9 ] ) );
assertTrue( isequaln( Ls, Le ) );

%% Errors
assert_throws( @() quadriga_lib.eval_antennas(), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.eval_antennas( L, [], 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( L ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.eval_antennas( 1 ), 'Input ''layout_in'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.eval_antennas( L ), 'the element spacing is given in wavelengths, but the layout has no frequency' );      % C++
assert_throws( @() quadriga_lib.eval_antennas( L, [ 3.5e9, 2e9 ] ), 'center_freq' );                                                     % C++

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.eval_antennas( varargin{:} ); %#ok<ASGLU>
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

% Write a text file; escapes such as \n in the text are converted by fprintf (the texts hold no percent signs)
function write_text(fn, text)
fid = fopen(fn, 'w');
fprintf(fid, text);
fclose(fid);
end

% Delete the files that exist
function delete_files(files)
for n = 1 : numel(files)
    if exist(files{n}, 'file')
        delete(files{n});
    end
end
end
