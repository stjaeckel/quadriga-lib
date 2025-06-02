function test_arrayant_qdant_write

%% Simple test
if exist( 'testm.qdant','file' )
    delete('testm.qdant');
end

azimuth_grid    = [-1.5,0,1.5,2] * pi/2;
elevation_grid  = [-0.9,0,0.9] * pi/2;
e_theta_re      = reshape(1:12,3,[])/2;
e_theta_im      = -reshape(1:12,3,[])*0.002;
e_phi_re        = -reshape(1:12,3,[]);
e_phi_im        = -reshape(1:12,3,[])*0.001;
element_pos     = [1;2;4];
coupling_re     = 1;
coupling_im     = 0.1;
center_freq     = 2e9;
name            = 'name';

id_file = quadriga_lib.arrayant_qdant_write( 'testm.qdant', [], [], [], e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
    azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, center_freq, name );

assert( id_file == 1 );

[e_theta_reI, e_theta_imI, e_phi_reI, e_phi_imI, azimuth_gridI, elevation_gridI, element_posI, ...
    coupling_reI, coupling_imI, center_frequencyI, nameI, layout] = quadriga_lib.arrayant_qdant_read('testm.qdant');

assertElementsAlmostEqual( azimuth_grid, azimuth_gridI, 'absolute', 1e-6 );
assertElementsAlmostEqual( elevation_grid, elevation_gridI, 'absolute', 1e-6 );
assertElementsAlmostEqual( e_theta_reI, e_theta_re, 'absolute', 1e-4 );
assertElementsAlmostEqual( e_theta_imI, e_theta_im, 'absolute', 1e-4 );
assertElementsAlmostEqual( e_phi_reI, e_phi_re, 'absolute', 1e-4 );
assertElementsAlmostEqual( e_phi_imI, e_phi_im, 'absolute', 1e-4 );
assertElementsAlmostEqual( element_posI, element_pos, 'absolute', 1e-4 );
assertElementsAlmostEqual( coupling_reI, coupling_re, 'absolute', 1e-4 );
assertElementsAlmostEqual( coupling_imI, coupling_im, 'absolute', 1e-4 );
assertElementsAlmostEqual( center_frequencyI, center_freq, 'absolute', 1e-4 );
assertTrue( strcmp(nameI,name) );
assertEqual( layout, uint32(1) );

delete('testm.qdant');

%% Complex test

f = fopen( 'testm.qdant','w' );
fprintf(f,'%s\n','<qdant><arrayant>');
fprintf(f,'%s\n','<name>bla</name>');
fprintf(f,'%s\n','<ElevationGrid>-90 -45 0 45 90</ElevationGrid>');
fprintf(f,'%s\n','<AzimuthGrid>-180 0 90</AzimuthGrid>');
fprintf(f,'%s\n',['<EthetaMag>',num2str(1:15),'</EthetaMag>']);
fprintf(f,'%s\n','</arrayant></qdant>');
fclose(f);

ant = struct;
ant.azimuth_grid    = [-1.5,0,1.5,2] * pi/2;
ant.elevation_grid  = [-0.9,0,0.9] * pi/2;
ant.e_theta_re      = reshape(1:12,3,[])/2;
ant.e_theta_im      = -reshape(1:12,3,[])*0.002;
ant.e_phi_re        = -reshape(1:12,3,[]);
ant.e_phi_im        = -reshape(1:12,3,[])*0.001;
ant.element_pos     = [1;2;4];
ant.coupling_re     = 1;
ant.coupling_im     = 0.1;
ant.center_freq     = 2e9;
ant.name            = 'name';

id_file = quadriga_lib.arrayant_qdant_write( 'testm.qdant', ant );

assertEqual( id_file, uint32(2) );

[A,B,C,D,E,F,G,H,I,J, nameI, layout] = quadriga_lib.arrayant_qdant_read('testm.qdant',2);

assertElementsAlmostEqual( A, ant.e_theta_re, 'absolute', 1e-4 );
assertElementsAlmostEqual( B, ant.e_theta_im, 'absolute', 1e-4 );
assertElementsAlmostEqual( C, ant.e_phi_re, 'absolute', 1e-4 );
assertElementsAlmostEqual( D, ant.e_phi_im, 'absolute', 1e-4 );
assertElementsAlmostEqual( E, ant.azimuth_grid, 'absolute', 1e-6 );
assertElementsAlmostEqual( F, ant.elevation_grid, 'absolute', 1e-6 );
assertElementsAlmostEqual( G, ant.element_pos, 'absolute', 1e-4 );
assertElementsAlmostEqual( H, ant.coupling_re, 'absolute', 1e-4 );
assertElementsAlmostEqual( I, ant.coupling_im, 'absolute', 1e-4 );
assertElementsAlmostEqual( J, ant.center_freq, 'absolute', 1e-4 );
assertTrue( strcmp(nameI,ant.name) );
assertEqual(layout, uint32([1,2]));

[~,~,~,~,~,~,~,~,~,~, nameI] = quadriga_lib.arrayant_qdant_read('testm.qdant');
assertTrue( strcmp(nameI,'bla') );

id_file = quadriga_lib.arrayant_qdant_write('testm.qdant', ant, 112);

assertEqual( id_file, uint32(112) );

layout = uint32([1,2,112 ; 112, 112, 6]);

% Try setting a layout that contains an ID that is not in the file
try
    id_file = quadriga_lib.arrayant_qdant_write('testm.qdant', ant, 5, layout);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Layout contains reference to non-existing array antenna!';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

quadriga_lib.arrayant_qdant_write('testm.qdant', ant, 6, layout);

% Check if the layout can be restored
[~,~,~,~,~,~,~,~,~,~, ~, layoutI] = quadriga_lib.arrayant_qdant_read('testm.qdant');
assertEqual(layout,layoutI);

% Test error: multiple outputs
try
    [~, ~] = quadriga_lib.arrayant_qdant_write('testm.qdant', ant, 999);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Test error: too few inputs
ant = struct;
ant.azimuth_grid    = [-1.5,0,1.5,2] * pi/2;
ant.elevation_grid  = [-0.9,0,0.9] * pi/2;
ant.e_theta_re      = reshape(1:12,3,[])/2;
ant.e_theta_im      = -reshape(1:12,3,[])*0.002;
try
    quadriga_lib.arrayant_qdant_write('testm.qdant', ant);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Field ''e_phi_re'' not found!';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    quadriga_lib.arrayant_qdant_write('testm.qdant', [],[],[], e_theta_re, e_theta_im, e_phi_re, e_phi_im);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Test error: elevation_grid mismatch
try
    id_file = quadriga_lib.arrayant_qdant_write( 'testm.qdant', [], [], [], e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
        azimuth_grid, elevation_grid(1:end-1));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of elements in ''elevation_grid'' does not match number of rows in pattern data.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Test error: azimuth_grid mismatch
try
    quadriga_lib.arrayant_qdant_write('testm.qdant', [],[],[], e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
        azimuth_grid(1:end-1), elevation_grid);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of elements in ''azimuth_grid'' does not match number of columns in pattern data.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Test error: pattern mismatch
try
    quadriga_lib.arrayant_qdant_write('testm.qdant', [],[],[], e_theta_re, e_theta_im(:,:,[1 1]), e_phi_re, e_phi_im, ...
        azimuth_grid, elevation_grid);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Sizes of ''e_theta_re'' and ''e_theta_im'' do not match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

quadriga_lib.arrayant_qdant_write('testm.qdant', [],[],[], e_theta_re, single(e_theta_im), e_phi_re, e_phi_im, azimuth_grid, elevation_grid);

delete('testm.qdant');
