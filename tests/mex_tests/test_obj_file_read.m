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
assertEqual( size(mtl_prop), [12,1] );
assertEqual( size(vert_list), [8,3] );
assertEqual( size(face_ind), [12,3] );
assertEqual( size(obj_ind), [12,1] );
assertEqual( size(mtl_ind), [12,1] );
assertEqual( size(obj_names), [1,1] );
assertTrue( isempty(mtl_names) );

assertTrue( isa(mesh, "double") );
assertTrue( isa(mtl_prop, "double") );
assertTrue( isa(vert_list, "double") );
assertTrue( isa(face_ind, "uint64") );
assertTrue( isa(obj_ind, "uint64") );
assertTrue( isa(mtl_ind, "uint64") );

assertTrue( all( mtl_prop(:,1) == 1 ) );    % a = 1 for vacuum (other cols cropped at defaults)

assertElementsAlmostEqual( vert_list, vert_list_correct, 'absolute', 1e-14 );
assertEqual( face_ind, uint64(face_ind_correct) );
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
[ mesh, mtl_prop, vert_list, face_ind, obj_ind, mtl_ind, obj_names, mtl_names ] = quadriga_lib.obj_file_read(fn);

assertEqual( size(obj_names), [2,1] );
assertEqual( size(mtl_names), [1,1] );

assertTrue( isa(mesh, "double") );
assertTrue( isa(mtl_prop, "double") );
assertTrue( isa(vert_list, "double") );

assertTrue( all( mtl_prop([1,2],1) == 1 ) );
assertTrue( all( mtl_prop([3,4],1) > 1.5 ) );
assertEqual( face_ind, uint64([2,3,1;2,4,3;6,7,5;6,8,7]) );
assertEqual( obj_ind, uint64([1;1;2;2]) );
assertEqual( mtl_ind, uint64([0;0;1;1]) );

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

[ mesh, mtl_prop, vert_list, face_ind, obj_ind, mtl_ind, obj_names, mtl_names, bsdf ] = quadriga_lib.obj_file_read(fn);

assertElementsAlmostEqual( mtl_prop(1,:), [1.1, 1.2, 1.3, 1.4, 10], 'absolute', 1e-14 );
assertElementsAlmostEqual( mtl_prop(2,:), [2.1, 2.2, 2.3, 2.4, 20], 'absolute', 1e-14 );

assertEqual( mtl_names{1,1}, 'Cst::1.1:1.2:1.3:1.4:10' );
assertEqual( mtl_names{2,1}, 'Cst::2.1:2.2:2.3:2.4:20' );
assertEqual( obj_ind, uint64([1;1]) );
assertEqual( mtl_ind, uint64([1;2]) );
assertEqual( size(bsdf), [0,0] );

% trim = false keeps all 16 columns
[~, mtl_prop_full] = quadriga_lib.obj_file_read(fn, '', false);
assertEqual( size(mtl_prop_full), [2, 16] );
assertElementsAlmostEqual( mtl_prop_full(1,:), [1.1, 1.2, 1.3, 1.4, 10, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'absolute', 1e-14 );

% Too many inputs
try
    quadriga_lib.obj_file_read(fn,'bla',true,99);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Too many outputs
try
    [~,~,~,~,~,~,~,~,~,~] = quadriga_lib.obj_file_read(fn);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Missing file name
try
    quadriga_lib.obj_file_read;
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Wrong file name
try
    quadriga_lib.obj_file_read('bla.obj');
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Error opening file: ''bla.obj'' does not exist.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

delete(fn);

if 0
    % Test Custom Materials CSV
    csv_fn = 'custom_materials.csv';

    % Test 1: Basic custom materials
    f = fopen(csv_fn, 'w');
    fprintf(f, '%s\n', 'name,a,b,c,d,att');
    fprintf(f, '%s\n', 'custom_material_1,2.5,0.0,0.001,0.5,5.0');
    fprintf(f, '%s\n', 'custom_material_2,4.0,-0.1,0.05,1.2,10.0');
    fclose(f);

    f = fopen(fn, 'w');
    fprintf(f, '%s\n', 'o Cube');
    fprintf(f, '%s\n', 'v 1.0 1.0 1.0');
    fprintf(f, '%s\n', 'v 1.0 1.0 -1.0');
    fprintf(f, '%s\n', 'v 1.0 -1.0 1.0');
    fprintf(f, '%s\n', 'v 1.0 -1.0 -1.0');
    fprintf(f, '%s\n', 'v -1.0 1.0 1.0');
    fprintf(f, '%s\n', 'v -1.0 1.0 -1.0');
    fprintf(f, '%s\n', 'v -1.0 -1.0 1.0');
    fprintf(f, '%s\n', 'v -1.0 -1.0 -1.0');
    fprintf(f, '%s\n', 'usemtl custom_material_1');
    fprintf(f, '%s\n', 'f 5 3 1');
    fprintf(f, '%s\n', 'f 3 8 4');
    fprintf(f, '%s\n', 'f 7 6 8');
    fprintf(f, '%s\n', 'f 2 8 6');
    fprintf(f, '%s\n', 'usemtl custom_material_2');
    fprintf(f, '%s\n', 'f 1 4 2');
    fprintf(f, '%s\n', 'f 5 2 6');
    fprintf(f, '%s\n', 'f 5 7 3');
    fprintf(f, '%s\n', 'f 3 7 8');
    fprintf(f, '%s\n', 'f 7 5 6');
    fprintf(f, '%s\n', 'f 2 4 8');
    fprintf(f, '%s\n', 'f 1 3 4');
    fprintf(f, '%s\n', 'f 5 1 2');
    fclose(f);

    [~, mtl_prop, ~, ~, ~, mtl_ind, ~, mtl_names] = quadriga_lib.obj_file_read(fn, csv_fn);

    assertElementsAlmostEqual(mtl_prop(1,:), [2.5, 0.0, 0.001, 0.5, 5.0, 0, 0, 0, 1], 'absolute', 1e-14);
    assertEqual(mtl_names{1,1}, 'custom_material_1');
    assertEqual(mtl_ind(1), uint64(1));

    assertElementsAlmostEqual(mtl_prop(5,:), [4.0, -0.1, 0.05, 1.2, 10.0, 0, 0, 0, 1], 'absolute', 1e-14);
    assertEqual(mtl_names{2,1}, 'custom_material_2');
    assertEqual(mtl_ind(5), uint64(2));

    delete(fn);
    delete(csv_fn);

    % Test 2: Jumbled column order
    f = fopen(csv_fn, 'w');
    fprintf(f, '%s\n', 'att,d,c,b,a,name');
    fprintf(f, '%s\n', '5.0,0.5,0.001,0.0,2.5,custom_material_1');
    fprintf(f, '%s\n', '10.0,1.2,0.05,-0.1,4.0,custom_material_2');
    fclose(f);

    f = fopen(fn, 'w');
    fprintf(f, '%s\n', 'o Cube');
    fprintf(f, '%s\n', 'v 1.0 1.0 1.0');
    fprintf(f, '%s\n', 'v 1.0 1.0 -1.0');
    fprintf(f, '%s\n', 'v 1.0 -1.0 1.0');
    fprintf(f, '%s\n', 'v 1.0 -1.0 -1.0');
    fprintf(f, '%s\n', 'v -1.0 1.0 1.0');
    fprintf(f, '%s\n', 'v -1.0 1.0 -1.0');
    fprintf(f, '%s\n', 'v -1.0 -1.0 1.0');
    fprintf(f, '%s\n', 'v -1.0 -1.0 -1.0');
    fprintf(f, '%s\n', 'usemtl custom_material_1');
    fprintf(f, '%s\n', 'f 5 3 1');
    fprintf(f, '%s\n', 'f 3 8 4');
    fprintf(f, '%s\n', 'f 7 6 8');
    fprintf(f, '%s\n', 'f 2 8 6');
    fprintf(f, '%s\n', 'usemtl custom_material_2');
    fprintf(f, '%s\n', 'f 1 4 2');
    fprintf(f, '%s\n', 'f 5 2 6');
    fprintf(f, '%s\n', 'f 5 7 3');
    fprintf(f, '%s\n', 'f 3 7 8');
    fprintf(f, '%s\n', 'f 7 5 6');
    fprintf(f, '%s\n', 'f 2 4 8');
    fprintf(f, '%s\n', 'f 1 3 4');
    fprintf(f, '%s\n', 'f 5 1 2');
    fclose(f);

    [~, mtl_prop, ~, ~, ~, ~, ~, mtl_names] = quadriga_lib.obj_file_read(fn, csv_fn);

    assertElementsAlmostEqual(mtl_prop(1,:), [2.5, 0.0, 0.001, 0.5, 5.0], 'absolute', 1e-14);
    assertEqual(mtl_names{1,1}, 'custom_material_1');

    assertElementsAlmostEqual(mtl_prop(5,:), [4.0, -0.1, 0.05, 1.2, 10.0], 'absolute', 1e-14);
    assertEqual(mtl_names{2,1}, 'custom_material_2');

    delete(fn);
    delete(csv_fn);

    % Test 3: Missing column - missing 'att'
    f = fopen(csv_fn, 'w');
    fprintf(f, '%s\n', 'name,a,b,c,d');
    fprintf(f, '%s\n', 'custom_material_1,2.5,0.0,0.001,0.5');
    fclose(f);

    f = fopen(fn, 'w');
    fprintf(f, '%s\n', 'o Cube');
    fprintf(f, '%s\n', 'v 1.0 1.0 1.0');
    fprintf(f, '%s\n', 'v 1.0 1.0 -1.0');
    fprintf(f, '%s\n', 'v 1.0 -1.0 1.0');
    fprintf(f, '%s\n', 'v 1.0 -1.0 -1.0');
    fprintf(f, '%s\n', 'usemtl custom_material_1');
    fprintf(f, '%s\n', 'f 1 2 3');
    fprintf(f, '%s\n', 'f 2 3 4');
    fclose(f);

    try
        quadriga_lib.obj_file_read(fn, csv_fn);
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    catch ME
        if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
            error('moxunit:exceptionNotRaised', 'Expected an error for missing column!');
        end
    end

    delete(fn);
    delete(csv_fn);

    % Test 4: Missing multiple columns
    f = fopen(csv_fn, 'w');
    fprintf(f, '%s\n', 'name,a,b');
    fprintf(f, '%s\n', 'custom_material_1,2.5,0.0');
    fclose(f);

    f = fopen(fn, 'w');
    fprintf(f, '%s\n', 'o Cube');
    fprintf(f, '%s\n', 'v 1.0 1.0 1.0');
    fprintf(f, '%s\n', 'v 1.0 1.0 -1.0');
    fprintf(f, '%s\n', 'v 1.0 -1.0 1.0');
    fprintf(f, '%s\n', 'v 1.0 -1.0 -1.0');
    fprintf(f, '%s\n', 'usemtl custom_material_1');
    fprintf(f, '%s\n', 'f 1 2 3');
    fprintf(f, '%s\n', 'f 2 3 4');
    fclose(f);

    try
        quadriga_lib.obj_file_read(fn, csv_fn);
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    catch ME
        if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
            error('moxunit:exceptionNotRaised', 'Expected an error for missing columns!');
        end
    end

    delete(fn);
    delete(csv_fn);

    % Test 5: Duplicate material names
    f = fopen(csv_fn, 'w');
    fprintf(f, '%s\n', 'name,a,b,c,d,att');
    fprintf(f, '%s\n', 'custom_material_1,2.5,0.0,0.001,0.5,5.0');
    fprintf(f, '%s\n', 'custom_material_1,4.0,-0.1,0.05,1.2,10.0');
    fclose(f);

    f = fopen(fn, 'w');
    fprintf(f, '%s\n', 'o Cube');
    fprintf(f, '%s\n', 'v 1.0 1.0 1.0');
    fprintf(f, '%s\n', 'v 1.0 1.0 -1.0');
    fprintf(f, '%s\n', 'v 1.0 -1.0 1.0');
    fprintf(f, '%s\n', 'v 1.0 -1.0 -1.0');
    fprintf(f, '%s\n', 'usemtl custom_material_1');
    fprintf(f, '%s\n', 'f 1 2 3');
    fprintf(f, '%s\n', 'f 2 3 4');
    fclose(f);

    try
        quadriga_lib.obj_file_read(fn, csv_fn);
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    catch ME
        if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
            error('moxunit:exceptionNotRaised', 'Expected an error for duplicate material names!');
        end
    end

    delete(fn);
    delete(csv_fn);

    % Test 6: Non-existent CSV file
    f = fopen(fn, 'w');
    fprintf(f, '%s\n', 'o Cube');
    fprintf(f, '%s\n', 'v 1.0 1.0 1.0');
    fprintf(f, '%s\n', 'v 1.0 1.0 -1.0');
    fprintf(f, '%s\n', 'v 1.0 -1.0 1.0');
    fprintf(f, '%s\n', 'v 1.0 -1.0 -1.0');
    fprintf(f, '%s\n', 'usemtl custom_material_1');
    fprintf(f, '%s\n', 'f 1 2 3');
    fprintf(f, '%s\n', 'f 2 3 4');
    fclose(f);

    try
        quadriga_lib.obj_file_read(fn, 'nonexistent.csv');
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    catch ME
        if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
            error('moxunit:exceptionNotRaised', 'Expected an error for non-existent CSV file!');
        end
    end

    % Full inline custom syntax: a:b:c:d:att:attB:alpha:alphaB:fRef
    delete(fn);
    f = fopen( fn,'w' );
    fprintf(f,'%s\n','o Plane');
    fprintf(f,'%s\n','v -1 -1 0');
    fprintf(f,'%s\n','v  1 -1 0');
    fprintf(f,'%s\n','v -1  1 0');
    fprintf(f,'%s\n','v  1  1 0');
    fprintf(f,'%s\n','usemtl Full::3.0:0.1:0.05:1.2:5.0:0.3:2.0:0.5:2.4');
    fprintf(f,'%s\n','f 1 2 3');
    fprintf(f,'%s\n','f 2 4 3');
    fclose(f);

    [~, mtl_prop, ~, ~, ~, ~, ~, mtl_names] = quadriga_lib.obj_file_read(fn);
    assertEqual( size(mtl_prop), [2,9] );
    assertElementsAlmostEqual( mtl_prop(1,:), [3.0, 0.1, 0.05, 1.2, 5.0, 0.3, 2.0, 0.5, 2.4], 'absolute', 1e-14 );
    assertElementsAlmostEqual( mtl_prop(2,:), mtl_prop(1,:), 'absolute', 1e-14 ); % same material, both rows identical
    assertEqual( mtl_names{1,1}, 'Full::3.0:0.1:0.05:1.2:5.0:0.3:2.0:0.5:2.4' );


    % Only a:b:c:d:att provided — attB, alpha, alphaB default to 0, fRef to 1
    delete(fn);
    f = fopen( fn,'w' );
    fprintf(f,'%s\n','o Plane');
    fprintf(f,'%s\n','v -1 -1 0');
    fprintf(f,'%s\n','v  1 -1 0');
    fprintf(f,'%s\n','v -1  1 0');
    fprintf(f,'%s\n','usemtl P5::1.5:0.0:0.01:0.8:3.0');
    fprintf(f,'%s\n','f 1 2 3');
    fclose(f);

    [~, mtl_prop] = quadriga_lib.obj_file_read(fn);
    assertElementsAlmostEqual( mtl_prop(1,:), [1.5, 0.0, 0.01, 0.8, 3.0, 0, 0, 0, 1], 'absolute', 1e-14 );

    % CSV with new optional columns (attB, alpha, alphaB, fRef)
    delete(fn);
    csv_fn = 'custom_materials.csv';

    f = fopen(csv_fn,'w');
    fprintf(f,'%s\n','name,a,b,c,d,att,attB,alpha,alphaB,fRef');
    fprintf(f,'%s\n','mat_full,3.0,0.1,0.05,1.2,5.0,0.3,2.0,0.5,2.4');
    fprintf(f,'%s\n','mat_partial,2.5,0.0,0.001,0.5,4.0');   % trailing cols missing → defaults
    fclose(f);

    f = fopen(fn,'w');
    fprintf(f,'%s\n','o Obj');
    fprintf(f,'%s\n','v -1 -1 0');
    fprintf(f,'%s\n','v  1 -1 0');
    fprintf(f,'%s\n','v -1  1 0');
    fprintf(f,'%s\n','usemtl mat_full');
    fprintf(f,'%s\n','f 1 2 3');
    fprintf(f,'%s\n','usemtl mat_partial');
    fprintf(f,'%s\n','f 1 3 2');
    fclose(f);

    [~, mtl_prop, ~, ~, ~, ~, ~, mtl_names] = quadriga_lib.obj_file_read(fn, csv_fn);
    assertEqual( size(mtl_prop), [2,9] );
    assertElementsAlmostEqual( mtl_prop(1,:), [3.0, 0.1, 0.05, 1.2, 5.0, 0.3, 2.0, 0.5, 2.4], 'absolute', 1e-14 );
    assertElementsAlmostEqual( mtl_prop(2,:), [2.5, 0.0, 0.001, 0.5, 4.0, 0, 0, 0, 1], 'absolute', 1e-14 );
    assertEqual( mtl_names{1,1}, 'mat_full' );
    assertEqual( mtl_names{2,1}, 'mat_partial' );
    delete(csv_fn);

    % CSV with only required columns
    f = fopen(csv_fn,'w');
    fprintf(f,'%s\n','name,a');
    fprintf(f,'%s\n','bare_mat,4.5');
    fclose(f);

    f = fopen(fn,'w');
    fprintf(f,'%s\n','o Obj');
    fprintf(f,'%s\n','v -1 -1 0');
    fprintf(f,'%s\n','v  1 -1 0');
    fprintf(f,'%s\n','v -1  1 0');
    fprintf(f,'%s\n','usemtl bare_mat');
    fprintf(f,'%s\n','f 1 2 3');
    fclose(f);

    [~, mtl_prop] = quadriga_lib.obj_file_read(fn, csv_fn);
    assertElementsAlmostEqual( mtl_prop(1,:), [4.5, 0, 0, 0, 0, 0, 0, 0, 1], 'absolute', 1e-14 );
    delete(csv_fn);

    % Non-unity fRef scales correctly in CSV
    % fRef = 2.4 GHz stored in mtl_prop col 9
    f = fopen(csv_fn,'w');
    fprintf(f,'%s\n','name,a,c,fRef');
    fprintf(f,'%s\n','wifi_wall,5.0,0.02,2.4');
    fclose(f);

    f = fopen(fn,'w');
    fprintf(f,'%s\n','o Obj');
    fprintf(f,'%s\n','v -1 -1 0');
    fprintf(f,'%s\n','v  1 -1 0');
    fprintf(f,'%s\n','v -1  1 0');
    fprintf(f,'%s\n','usemtl wifi_wall');
    fprintf(f,'%s\n','f 1 2 3');
    fclose(f);

    [~, mtl_prop] = quadriga_lib.obj_file_read(fn, csv_fn);
    assertElementsAlmostEqual( mtl_prop(1,1), 5.0,  'absolute', 1e-14 ); % a
    assertElementsAlmostEqual( mtl_prop(1,3), 0.02, 'absolute', 1e-14 ); % c
    assertElementsAlmostEqual( mtl_prop(1,9), 2.4,  'absolute', 1e-14 ); % fRef
    delete(fn); delete(csv_fn);


    delete(fn);
end
end
