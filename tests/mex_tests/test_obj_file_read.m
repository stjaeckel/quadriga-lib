function test_obj_file_read
% Delete file
fn = 'cube.obj';
if exist( fn,'file' )
    delete(fn);
end

f = fopen( fn,'w' );
fprintf(f,'%s\n','# A very nice, but useless comment ;-)');
fprintf(f,'%s\n','o Cube');
fprintf(f,'%s\n','v 1.0 1.0 1.0');
fprintf(f,'%s\n','v 1.0 1.0 -1.0');
fprintf(f,'%s\n','v 1.0 -1.0 1.0');
fprintf(f,'%s\n','v 1.0 -1.0 -1.0');
fprintf(f,'%s\n','v -1.0 1.0 1.0');
fprintf(f,'%s\n','v -1.0 1.0 -1.0');
fprintf(f,'%s\n','v -1.0 -1.0 1.0');
fprintf(f,'%s\n','v -1.0 -1.0 -1.0');
fprintf(f,'%s\n','s 0');
fprintf(f,'%s\n','f 5 3 1');
fprintf(f,'%s\n','f 3 8 4');
fprintf(f,'%s\n','f 7 6 8');
fprintf(f,'%s\n','f 2 8 6');
fprintf(f,'%s\n','f 1 4 2');
fprintf(f,'%s\n','f 5 2 6');
fprintf(f,'%s\n','f 5 7 3');
fprintf(f,'%s\n','f 3 7 8');
fprintf(f,'%s\n','f 7 5 6');
fprintf(f,'%s\n','f 2 4 8');
fprintf(f,'%s\n','f 1 3 4');
fprintf(f,'%s\n','f 5 1 2');
fclose(f);

vert_list_correct = [1.0 1.0 1.0; 1.0 1.0 -1.0; 1.0 -1.0 1.0; 1.0 -1.0 -1.0; -1.0 1.0 1.0; -1.0 1.0 -1.0'; -1.0 -1.0 1.0; -1.0 -1.0 -1.0 ];
face_ind_correct  = [ 5 3 1;  3 8 4; 7 6 8;  2 8 6;    1 4 2;  5 2 6;  5 7 3;  3 7 8;    7 5 6;  2 4 8;  1 3 4;  5 1 2 ];
mesh_correct = [ vert_list_correct( face_ind_correct(:,1),: ), vert_list_correct( face_ind_correct(:,2),: ), vert_list_correct( face_ind_correct(:,3),: ) ];

% No output should be fine
quadriga_lib.obj_file_read(fn);

% Read all
[ mesh, mtl_prop, vert_list, face_ind, obj_ind, mtl_ind, obj_names, mtl_names ] = quadriga_lib.obj_file_read(fn);

assertEqual( size(mesh), [12,9] );
assertEqual( size(mtl_prop), [12,5] );
assertEqual( size(vert_list), [8,3] );
assertEqual( size(face_ind), [12,3] );
assertEqual( size(obj_ind), [12,1] );
assertEqual( size(mtl_ind), [12,1] );
assertEqual( size(obj_names), [1,1] );
assertTrue( isempty(mtl_names) );

assertTrue( isa(mesh, "double") );
assertTrue( isa(mtl_prop, "double") );
assertTrue( isa(vert_list, "double") );
assertTrue( isa(face_ind, "uint32") );
assertTrue( isa(obj_ind, "uint32") );
assertTrue( isa(mtl_ind, "uint32") );

assertTrue( all( mtl_prop(:,1) == 1 ) );
assertTrue( all(all( mtl_prop(:,[2,3,4,5]) == 0 )) );

assertElementsAlmostEqual( vert_list, vert_list_correct, 'absolute', 1e-14 );
assertEqual( face_ind, uint32(face_ind_correct) );
assertElementsAlmostEqual( mesh, mesh_correct, 'absolute', 1e-14 );

assertTrue( all( obj_ind == 1 ) );
assertTrue( all( mtl_ind == 0 ) );
assertEqual( obj_names{1,1}, 'Cube' );

% Two planes
delete(fn);

f = fopen( fn,'w' );
fprintf(f,'%s\n','o Plane');
fprintf(f,'%s\n','v -1.000000 -1.000000 0.000000');
fprintf(f,'%s\n','v 1.000000 -1.000000 0.000000');
fprintf(f,'%s\n','v -1.000000 1.000000 0.000000');
fprintf(f,'%s\n','v 1.000000 1.000000 0.000000');
fprintf(f,'%s\n','vt 1.000000 0.000000');
fprintf(f,'%s\n','vt 0.000000 1.000000');
fprintf(f,'%s\n','vt 0.000000 0.000000');
fprintf(f,'%s\n','vt 1.000000 1.000000');
fprintf(f,'%s\n','f 2/1 3/2 1/3');
fprintf(f,'%s\n','f 2/1 4/4 3/2');
fprintf(f,'%s\n','o Plane.001');
fprintf(f,'%s\n','v -1.000000 -1.000000 1.26');
fprintf(f,'%s\n','v 1.000000 -1.000000 1.26');
fprintf(f,'%s\n','v -1.000000 1.000000 1.26');
fprintf(f,'%s\n','v 1.000000 1.000000 1.26');
fprintf(f,'%s\n','vt 1.000000 0.000000');
fprintf(f,'%s\n','vt 0.000000 1.000000');
fprintf(f,'%s\n','vt 0.000000 0.000000');
fprintf(f,'%s\n','vt 1.000000 1.000000');
fprintf(f,'%s\n','usemtl itu_wood');
fprintf(f,'%s\n','f 6/5 7/6 5/7');
fprintf(f,'%s\n','f 6/5 8/8 7/6');
fclose(f);

% Read all
[ mesh, mtl_prop, vert_list, face_ind, obj_ind, mtl_ind, obj_names, mtl_names ] = quadriga_lib.obj_file_read(fn,1);

assertEqual( size(obj_names), [2,1] );
assertEqual( size(mtl_names), [1,1] );

assertTrue( isa(mesh, "single") );
assertTrue( isa(mtl_prop, "single") );
assertTrue( isa(vert_list, "single") );

assertTrue( all( mtl_prop([1,2],1) == 1 ) );
assertTrue( all( mtl_prop([3,4],1) > 1.5 ) );
assertEqual( face_ind, uint32([2,3,1;2,4,3;6,7,5;6,8,7]) );
assertEqual( obj_ind, uint32([1;1;2;2]) );
assertEqual( mtl_ind, uint32([0;0;1;1]) );

assertEqual( mtl_names{1,1}, 'itu_wood' );

% Custom Material

delete(fn);
f = fopen( fn,'w' );
fprintf(f,'%s\n','o Plane');
fprintf(f,'%s\n','v -1.000000 -1.000000 0.000000');
fprintf(f,'%s\n','v 1.000000 -1.000000 0.000000');
fprintf(f,'%s\n','v -1.000000 1.000000 0.000000');
fprintf(f,'%s\n','v 1.000000 1.000000 0.000000');
fprintf(f,'%s\n','vn -0.0000 -0.0000 1.0000');
fprintf(f,'%s\n','vt 1.000000 0.000000');
fprintf(f,'%s\n','vt 0.000000 1.000000');
fprintf(f,'%s\n','vt 0.000000 0.000000');
fprintf(f,'%s\n','vt 1.000000 1.000000');
fprintf(f,'%s\n','usemtl Cst::1.1:1.2:1.3:1.4:10');
fprintf(f,'%s\n','f 2/1/1 3/2/1 1/3/1');
fprintf(f,'%s\n','usemtl Cst::2.1:2.2:2.3:2.4:20');
fprintf(f,'%s\n','f 2/1/1 4/4/1 3/2/1');
fclose(f);

[ mesh, mtl_prop, vert_list, face_ind, obj_ind, mtl_ind, obj_names, mtl_names ] = quadriga_lib.obj_file_read(fn,0);
assertTrue( isa(mesh, "double") );

assertElementsAlmostEqual( mtl_prop(1,:), [1.1, 1.2, 1.3, 1.4, 10], 'absolute', 1e-14 );
assertElementsAlmostEqual( mtl_prop(2,:), [2.1, 2.2, 2.3, 2.4, 20], 'absolute', 1e-14 );
assertEqual( mtl_names{1,1}, 'Cst::1.1:1.2:1.3:1.4:10' );
assertEqual( mtl_names{2,1}, 'Cst::2.1:2.2:2.3:2.4:20' );
assertEqual( obj_ind, uint32([1;1]) );
assertEqual( mtl_ind, uint32([1;2]) );

% Too many inputs
try
    quadriga_lib.obj_file_read(fn,0,1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Too many input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Too many outputs
try
    [~,~,~,~,~,~,~,~,~] = quadriga_lib.obj_file_read(fn,0);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Too many output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Missing file name
try
    quadriga_lib.obj_file_read;
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Filename is missing.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Wrong file name
try
    quadriga_lib.obj_file_read('bla.obj');
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Error opening file.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

delete(fn);

end
