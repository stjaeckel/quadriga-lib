% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_attach_terminals
% MEX interface test for quadriga_lib.attach_terminals.
% Scope: argument and output handling, 1-based node and site indices, defaults, error paths.
% The two rules, the ties and the antennas are tested in the Catch2 suite (test_attach_terminals.cpp).

%% Setup: site A (100 m from U1) with two sectors, site B (100 m from U2); UMa
fn_tx = 'test_attach_terminals_tx.csv';
fn_rx = 'test_attach_terminals_rx.csv';
cleanup = onCleanup( @() delete_files( { fn_tx, fn_rx } ) ); %#ok<NASGU>
write_text( fn_tx, [ 'name,sector,x,y,z,scenario,heading\n', ...
    'A,0,0,0,25,UMa,0\n', ...
    'A,1,,,,,180\n', ...
    'B,0,1000,0,25,UMa,0\n' ] );
write_text( fn_rx, 'name,x,y,z,indoor\nU1,100,0,1.5,0\nU2,900,0,1.5,0\n' );
L = quadriga_lib.layout_csv_read( fn_tx, fn_rx, [ 2e9, 3.5e9 ] );
L2 = quadriga_lib.init_link_states( L, '38.901', [], false, 2e9, '', 5 );

%% Basic call: the serving Tx node and site of each terminal, 1-based; the coupling gain of all Tx nodes
[ node, site, gain_dB ] = quadriga_lib.attach_terminals( L2 );
assertEqual( node, uint64( [ 1; 3 ] ) );                                        % a tie of the sectors of A goes to the lowest node
assertEqual( site, uint64( [ 1; 2 ] ) );
assertEqual( size( gain_dB ), [ 2, 3 ] );                                       % [n_rx_terminal, n_tx_node]
assertEqual( gain_dB(:, 1), gain_dB(:, 2) );                                    % same antenna at the same position
assertTrue( gain_dB(1, 1) > gain_dB(1, 3) && gain_dB(2, 3) > gain_dB(2, 1) );   % the nearer site has the higher gain

%% Optional inputs: rule 'rsrp' with the rays of all links, frequency and port (1-based)
rays = quadriga_lib.generate_rays( L2 );
[ node2, site2, rsrp ] = quadriga_lib.attach_terminals( L2, 'rsrp', rays );
assertEqual( node2, node );
assertEqual( site2, site );
assertEqual( size( rsrp ), [ 2, 3 ] );
assertFalse( isequal( rsrp, gain_dB ) );
[ node3, ~, gain3 ] = quadriga_lib.attach_terminals( L2, 'large_scale', {}, 2, 2 );
assertEqual( node3, node );
assertFalse( isequal( gain3, gain_dB ) );                                       % another band and port
[ node4, ~, gain4 ] = quadriga_lib.attach_terminals( L2, 'large_scale', [], 1, 1 );
assertEqual( node4, node );
assertEqual( gain4, gain_dB );                                                  % [] and the defaults
[ ~, ~, gain5 ] = quadriga_lib.attach_terminals( L2, 'rsrp', rays, 2 );
assertFalse( isequal( gain5, rsrp ) );
L3 = L2;
L3.pairing = [ 2; 1 ];                                                          % pairing is ignored
assertEqual( quadriga_lib.attach_terminals( L3 ), node );
assertEqual( L2.pairing, zeros( 0, 0, 'uint64' ) );

%% Outputs: none and fewer
quadriga_lib.attach_terminals( L2 );
assertEqual( ans, node );
[ node6, site6 ] = quadriga_lib.attach_terminals( L2 );
assertEqual( node6, node );
assertEqual( site6, site );

%% Type handling: int8 frequency and port
assertEqual( quadriga_lib.attach_terminals( L2, 'large_scale', {}, int8( 2 ), uint8( 2 ) ), node );

%% Errors
assert_throws( @() quadriga_lib.attach_terminals(), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.attach_terminals( L2, 'rsrp', rays, 1, 1, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out4( L2 ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.attach_terminals( 1 ), 'Input ''layout'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.attach_terminals( L2, 'large_scale', 5 ), 'Input ''rays'' must be a cell array of struct arrays.' );
assert_throws( @() quadriga_lib.attach_terminals( L2, 'large_scale', {}, 0 ), 'Input ''i_freq'' cannot be 0 (1-based index).' );
assert_throws( @() quadriga_lib.attach_terminals( L2, 'large_scale', {}, 1, 0 ), 'Input ''tx_port'' cannot be 0 (1-based index).' );
assert_throws( @() quadriga_lib.attach_terminals( L2, 'large_scale', {}, 3 ), 'Input ''i_freq'' must select one of the 2 frequencies of the layout.' );   % C++
assert_throws( @() quadriga_lib.attach_terminals( L2, 'large_scale', {}, 1, 3 ), 'Input ''tx_port'' must select one of the 2 ports' );                        % C++
assert_throws( @() quadriga_lib.attach_terminals( L2, 'foo' ), 'Input ''rule'' must be ''large_scale'' or ''rsrp''.' );                                                 % C++
assert_throws( @() quadriga_lib.attach_terminals( L2, 'rsrp' ), 'Input ''rays'' must hold the rays of all links' );                                                  % C++

end

% Request 4 outputs so that the output-count guard fires
function out4(varargin)
[~, ~, ~, ~] = quadriga_lib.attach_terminals( varargin{:} ); %#ok<ASGLU>
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
