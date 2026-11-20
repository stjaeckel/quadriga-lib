% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_mitsuba_xml_file_write
% MEX interface test for quadriga_lib.mitsuba_xml_file_write.
% Scope: argument handling, index conversion, error paths.
% Physics validation lives in the Catch2 suite (test_mitsuba_xml_file_write.cpp).

%% Setup: a box (12 faces, concrete) and a wall (2 faces, wood), written as OBJ and read back
obj_fn = 'mitsuba_test_scene.obj';
mtl_fn = 'mitsuba_test_scene.mtl';
fn = 'mitsuba_test_scene.xml';
mesh_dir = 'mitsuba_test_scene_meshes';
cleanup_files( obj_fn, mtl_fn, fn, mesh_dir );

box = quadriga_lib.cube();
wall = quadriga_lib.plane( [], [], [5, 0, 0] );
n_box = size(box, 1);
n_wall = size(wall, 1);
quadriga_lib.obj_file_write( obj_fn, [box; wall], [ones(n_box,1); 2*ones(n_wall,1)], ...
    [ones(n_box,1); 2*ones(n_wall,1)], {'box', 'wall'}, {'concrete', 'wood'} );
[ ~, vert_list, face_ind, obj_ind, obj_names, mtl_ind, mtl_names, bsdf ] = quadriga_lib.obj_file_read( obj_fn );
cleanup_files( obj_fn, mtl_fn );

%% Basic call: one PLY per object with the faces and vertices of that object
quadriga_lib.mitsuba_xml_file_write( fn, vert_list, face_ind, obj_ind, mtl_ind, obj_names, mtl_names );
xml_ref = fileread( fn );
assertTrue( ~isempty(strfind(xml_ref, [mesh_dir, '/box.ply'])) );
assertTrue( ~isempty(strfind(xml_ref, [mesh_dir, '/wall.ply'])) );
assertTrue( ~isempty(strfind(xml_ref, 'concrete')) );
assertTrue( ~isempty(strfind(xml_ref, 'wood')) );
assertTrue( isempty(strfind(xml_ref, 'itu_')) );
assertTrue( isempty(strfind(xml_ref, 'principled')) );
box_vert = single( vert_list( unique(face_ind(obj_ind == 1, :)), : ) );   % PLY vertices: float, in index order
wall_vert = single( vert_list( unique(face_ind(obj_ind == 2, :)), : ) );
assertEqual( ply_read( fullfile(mesh_dir, 'box.ply') ), {box_vert, n_box} );
assertEqual( ply_read( fullfile(mesh_dir, 'wall.ply') ), {wall_vert, n_wall} );
box_ply = read_bytes( fullfile(mesh_dir, 'box.ply') );

%% Optional inputs: bsdf omitted equals [], given adds principled BSDFs; map_to_itu_materials
quadriga_lib.mitsuba_xml_file_write( fn, vert_list, face_ind, obj_ind, mtl_ind, obj_names, mtl_names, [] );
assertEqual( fileread(fn), xml_ref );
quadriga_lib.mitsuba_xml_file_write( fn, vert_list, face_ind, obj_ind, mtl_ind, obj_names, mtl_names, [], [] );
assertEqual( fileread(fn), xml_ref );
quadriga_lib.mitsuba_xml_file_write( fn, vert_list, face_ind, obj_ind, mtl_ind, obj_names, mtl_names, [], false );
assertEqual( fileread(fn), xml_ref );

quadriga_lib.mitsuba_xml_file_write( fn, vert_list, face_ind, obj_ind, mtl_ind, obj_names, mtl_names, bsdf );
assertTrue( ~isempty(strfind(fileread(fn), 'principled')) );

quadriga_lib.mitsuba_xml_file_write( fn, vert_list, face_ind, obj_ind, mtl_ind, obj_names, mtl_names, [], true );
xml = fileread( fn );
assertTrue( ~isempty(strfind(xml, 'itu_concrete')) );
assertTrue( ~isempty(strfind(xml, 'itu_wood')) );

%% Type handling and orientation: double and other integer classes, single, row vectors give the same files
quadriga_lib.mitsuba_xml_file_write( fn, single(vert_list), double(face_ind), uint8(obj_ind)', ...
    double(mtl_ind)', obj_names, mtl_names );
assertEqual( fileread(fn), xml_ref );
assertEqual( read_bytes( fullfile(mesh_dir, 'box.ply') ), box_ply );

%% Indices: mtl_ind passes through (1-based in C++)
% Material 2 on the box: its PLY is named box_wood after the split by material
quadriga_lib.mitsuba_xml_file_write( fn, vert_list, face_ind, obj_ind, [2; mtl_ind(2:end)], obj_names, mtl_names );
assertTrue( ~isempty(strfind(fileread(fn), [mesh_dir, '/box_wood.ply'])) );

%% Errors: argument counts
args = { fn, vert_list, face_ind, obj_ind, mtl_ind, obj_names, mtl_names };
assert_throws( @() quadriga_lib.mitsuba_xml_file_write( args{1:6} ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.mitsuba_xml_file_write( args{:}, [], false, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out1( args{:} ), 'Wrong number of output arguments.' );

%% Errors: 0 in converted indices, C++ checks
f0 = face_ind; f0(1) = 0;
assert_throws( @() quadriga_lib.mitsuba_xml_file_write( fn, vert_list, f0, obj_ind, mtl_ind, obj_names, mtl_names ), ...
    'Entries in ''face_ind'' cannot be 0 (1-based index).' );
assert_throws( @() quadriga_lib.mitsuba_xml_file_write( fn, vert_list, face_ind, 0 * obj_ind, mtl_ind, obj_names, mtl_names ), ...
    'Entries in ''obj_ind'' cannot be 0 (1-based index).' );
assert_throws( @() quadriga_lib.mitsuba_xml_file_write( fn, vert_list, face_ind, obj_ind, 0 * mtl_ind, obj_names, mtl_names ), ...
    'Input ''mtl_ind'' cannot contain 0' );
assert_throws( @() quadriga_lib.mitsuba_xml_file_write( fn, vert_list(:, 1:2), face_ind, obj_ind, mtl_ind, obj_names, mtl_names ), ...
    '3 columns' );
assert_throws( @() quadriga_lib.mitsuba_xml_file_write( fn, vert_list, face_ind, obj_ind, mtl_ind, obj_names(1), mtl_names ), ...
    '''obj_names''' );

cleanup_files( fn, mesh_dir );
end

% Vertices [n_vert, 3] (single) and number of faces of a binary PLY file
function out = ply_read( ply_fn )
b = read_bytes( ply_fn );
i0 = strfind( char(b), sprintf('end_header\n') ) + 11;   % first byte after the ASCII header
nv = regexp( char(b(1:i0)), 'element vertex (\d+)', 'tokens', 'once' );
nf = regexp( char(b(1:i0)), 'element face (\d+)', 'tokens', 'once' );
nv = str2double( nv{1} );
xyz = reshape( typecast( b(i0 : i0 + 12*nv - 1), 'single' ), 3, nv )';
out = { xyz, str2double(nf{1}) };
end

function b = read_bytes( fn )
fid = fopen( fn, 'r' );
b = fread( fid, Inf, 'uint8=>uint8' )';
fclose( fid );
end

function cleanup_files( varargin )
for i = 1:numel(varargin)
    if exist( varargin{i}, 'dir' ) == 7
        if exist( 'OCTAVE_VERSION', 'builtin' )
            confirm_recursive_rmdir( false, 'local' );   % Octave asks before a recursive rmdir
        end
        rmdir( varargin{i}, 's' );
    elseif exist( varargin{i}, 'file' ) == 2
        delete( varargin{i} );
    end
end
end

function out1(varargin)
x = quadriga_lib.mitsuba_xml_file_write( varargin{:} ); %#ok<NASGU>
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
