function test_qdant_read

%% Minimal test
f = fopen( 'test.qdant','w' );
fprintf(f,'%s\n','<qdant><arrayant>');
fprintf(f,'%s\n','<name>bla</name>');
fprintf(f,'%s\n','<ElevationGrid>-90 -45 0 45 90</ElevationGrid>');
fprintf(f,'%s\n','<AzimuthGrid>-180 0 90</AzimuthGrid>');
fprintf(f,'%s\n',['<EthetaMag>',num2str(1:15),'</EthetaMag>']);
fprintf(f,'%s\n','</arrayant></qdant>');
fclose(f);

[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_frequency, name, layout] = quadriga_lib.arrayant_qdant_read('test.qdant');

assertTrue( isa(e_theta_re,'double') );
assertTrue( isa(e_theta_im,'double') );
assertTrue( isa(e_phi_re,'double') );
assertTrue( isa(e_phi_im,'double') );
assertTrue( isa(azimuth_grid,'double') );
assertTrue( isa(elevation_grid,'double') );
assertTrue( isa(element_pos,'double') );
assertTrue( isa(coupling_re,'double') );
assertTrue( isa(coupling_im,'double') );
assertTrue( isa(center_frequency,'double') );
assertTrue( isa(layout,'uint32') );

assertElementsAlmostEqual( 20*log10(e_theta_re), reshape(1:15,3,5)', 'absolute', 1e-14 );
assertElementsAlmostEqual( e_theta_im, zeros(5,3), 'absolute', 1e-14 );
assertElementsAlmostEqual( e_phi_re, zeros(5,3), 'absolute', 1e-14 );
assertElementsAlmostEqual( e_phi_im, zeros(5,3), 'absolute', 1e-14 );
assertElementsAlmostEqual( azimuth_grid, [-pi,0,pi/2], 'absolute', 1e-13 );
assertElementsAlmostEqual( elevation_grid, [-pi/2,-pi/4,0,pi/4,pi/2], 'absolute', 1e-13 );
assertElementsAlmostEqual( element_pos, [0;0;0], 'absolute', 1e-13 );
assertElementsAlmostEqual( coupling_re, 1, 'absolute', 1e-13 );
assertElementsAlmostEqual( coupling_im, 0, 'absolute', 1e-13 );
assertElementsAlmostEqual( center_frequency, 299792448, 'absolute', 1e-13 );
assertTrue( strcmp(name,'bla') );
assertEqual( layout, uint32(1) );

%% More complex test
f = fopen( 'test.qdant','w' );
fprintf(f,'%s\n','<?xml version="1.0" encoding="UTF-8"?><qdant xmlns:xx="test">');
fprintf(f,'%s\n','<xx:layout>1,1 1,1 1,1</xx:layout>');
fprintf(f,'%s\n','<xx:arrayant id="1">');
fprintf(f,'%s\n','<xx:AzimuthGrid>-90 -45 0 45 90</xx:AzimuthGrid>');
fprintf(f,'%s\n','<xx:ElevationGrid>-90 0 90</xx:ElevationGrid>');
fprintf(f,'%s\n',['<xx:EphiMag>',num2str(zeros(1,15)),'</xx:EphiMag>']);
fprintf(f,'%s\n',['<xx:EphiPhase>',num2str(ones(1,15)*90),'</xx:EphiPhase>']);
fprintf(f,'%s\n',['<xx:EthetaMag>',num2str(ones(1,15)*3),'</xx:EthetaMag>']);
fprintf(f,'%s\n',['<xx:EthetaPhase>',num2str(ones(1,15)*-90),'</xx:EthetaPhase>']);
fprintf(f,'%s\n','<xx:ElementPosition>1,2,3</xx:ElementPosition>');
fprintf(f,'%s\n','<xx:CouplingAbs>1</xx:CouplingAbs>');
fprintf(f,'%s\n','<xx:CouplingPhase>45</xx:CouplingPhase>');
fprintf(f,'%s\n','<xx:CenterFrequency>3e9</xx:CenterFrequency>');
fprintf(f,'%s\n','</xx:arrayant></qdant>');
fclose(f);

[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_frequency, name, layout] = quadriga_lib.arrayant_qdant_read('test.qdant',1,1);

assertTrue( isa(e_theta_re,'single') );
assertTrue( isa(e_theta_im,'single') );
assertTrue( isa(e_phi_re,'single') );
assertTrue( isa(e_phi_im,'single') );
assertTrue( isa(azimuth_grid,'single') );
assertTrue( isa(elevation_grid,'single') );
assertTrue( isa(element_pos,'single') );
assertTrue( isa(coupling_re,'single') );
assertTrue( isa(coupling_im,'single') );
assertTrue( isa(center_frequency,'single') );

assertElementsAlmostEqual( azimuth_grid, [-pi/2,-pi/4,0,pi/4,pi/2], 'absolute', 1e-7 );
assertElementsAlmostEqual( elevation_grid, [-pi/2,0,pi/2], 'absolute', 1e-7 );
assertElementsAlmostEqual( e_theta_re, zeros(3,5), 'absolute', 1e-7 );
assertElementsAlmostEqual( e_theta_im, -sqrt(10^(0.3)*ones(3,5)), 'absolute', 1e-6 );
assertElementsAlmostEqual( e_phi_re, zeros(3,5), 'absolute', 1e-7 );
assertElementsAlmostEqual( e_phi_im, ones(3,5), 'absolute', 1e-7 );
assertElementsAlmostEqual( element_pos, [1;2;3], 'absolute', 1e-7 );
assertElementsAlmostEqual( coupling_re, 1/sqrt(2), 'absolute', 1e-7 );
assertElementsAlmostEqual( coupling_im, 1/sqrt(2), 'absolute', 1e-7 );
assertElementsAlmostEqual( center_frequency, 3e9, 'absolute', 1e-13 );
assertTrue( strcmp(name,'unknown') );
assertEqual( layout, ones(2,3,'uint32') );

%% Two array antennas with uncommon formats
f = fopen( 'test.qdant','w' );
fprintf(f,'<qdant><arrayant id="1">\n');
fprintf(f,'<ElevationGrid> -45 45</ElevationGrid>\n');
fprintf(f,'<AzimuthGrid>-90 0 90</AzimuthGrid>\n');
fprintf(f,'<EthetaMag>\n\t1 1 1\n\t2 2 3\n</EthetaMag>\n');
fprintf(f,'</arrayant><arrayant id="3">\n');
fprintf(f,'<ElevationGrid> -45 45</ElevationGrid>');
fprintf(f,'<NoElements>2</NoElements>');
fprintf(f,'<AzimuthGrid>-90 0 90</AzimuthGrid>\n');
fprintf(f,'<EphiMag el="2">\n\t1 2 3\n\t-1 -2 -3\n</EphiMag>\n');
fprintf(f,'<EphiPhase el="2">90 90 90 -90 -90 -90</EphiPhase>\n');
fprintf(f,'<CouplingAbs>1 2</CouplingAbs>\n');
fprintf(f,'<CouplingPhase>45 -90</CouplingPhase>\n');
fprintf(f,'<name>xxx</name>\n');
fprintf(f,'</arrayant></qdant>\n');
fclose(f);

[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_frequency, name, layout] = quadriga_lib.arrayant_qdant_read('test.qdant',1);

assertEqual( layout, uint32([1 3]) );

assertElementsAlmostEqual( e_theta_re, sqrt([1.26 1.26 1.26 ; 1.58 1.58 2]), 'absolute', 1e-2 );

[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_frequency, name] = quadriga_lib.arrayant_qdant_read('test.qdant',3);

assertElementsAlmostEqual( e_theta_re, zeros(2,3,2), 'absolute', 1e-13 );
assertElementsAlmostEqual( e_phi_re, zeros(2,3,2), 'absolute', 1e-13 );
assertElementsAlmostEqual( e_phi_im(:,:,2), [1;-1].*sqrt(10.^([1 2 3 ; -1 -2 -3]/10)), 'absolute', 1e-13 );
assertElementsAlmostEqual( coupling_re, [1/sqrt(2);0], 'absolute', 1e-13 );
assertElementsAlmostEqual( coupling_im, [1/sqrt(2);-2], 'absolute', 1e-13 );
assertTrue( strcmp(name,'xxx') );

%% Check error parsing
try
    [~,~,~,~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_qdant_read('test.qdant',3);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
end

% AzimuthGrid missing
f = fopen( 'test.qdant','w' );
fprintf(f,'<qdant><arrayant>\n');
fprintf(f,'<ElevationGrid>-90 -45 0 45 90</ElevationGrid>\n');
fprintf(f,'</arrayant></qdant>');
fclose(f);

try
    [~,~,~,~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_qdant_read('test.qdant');
    error('moxunit:exceptionNotRaised', 'Expected an error!');
end

% AzimuthGrid tag not closed correctly
f = fopen( 'test.qdant','w' );
fprintf(f,'<qdant><arrayant>\n');
fprintf(f,'<ElevationGrid>-90 -45 0 45 90</ElevationGrid>\n');
fprintf(f,'<AzimuthGrid>-90 0 90</AzimuthGridXXX>\n');
fprintf(f,'</arrayant></qdant>');
fclose(f);

try
    [~,~,~,~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_qdant_read('test.qdant');
    error('moxunit:exceptionNotRaised', 'Expected an error!');
end

% Wrong number of entries in EthetaMag
f = fopen( 'test.qdant','w' );
fprintf(f,'<qdant><arrayant>\n');
fprintf(f,'<ElevationGrid>-90 -45 0 45 90</ElevationGrid>\n');
fprintf(f,'<AzimuthGrid>-90 0 90</AzimuthGrid>\n');
fprintf(f,'%s\n',['<EthetaMag>',num2str(1:16),'</EthetaMag>']);
fprintf(f,'</arrayant></qdant>');
fclose(f);

try
    [~,~,~,~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_qdant_read('test.qdant');
    error('moxunit:exceptionNotRaised', 'Expected an error!');
end

% Wrong number of entries in CouplingAbs
f = fopen( 'test.qdant','w' );
fprintf(f,'<qdant><arrayant><NoElements>2</NoElements>\n');
fprintf(f,'<ElevationGrid>-90 -45 0 45 90</ElevationGrid>\n');
fprintf(f,'<AzimuthGrid>-90 0 90</AzimuthGrid>\n');
fprintf(f,'%s\n',['<EthetaMag>',num2str(1:15),'</EthetaMag>']);
fprintf(f,'<CouplingAbs>1 2 3</CouplingAbs>\n');
fprintf(f,'</arrayant></qdant>');
fclose(f);

try
    [~,~,~,~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_qdant_read('test.qdant');
    error('moxunit:exceptionNotRaised', 'Expected an error!');
end

% CouplingPhase with multiple ports without CouplingAbs
f = fopen( 'test.qdant','w' );
fprintf(f,'<qdant><arrayant>\n');
fprintf(f,'<ElevationGrid>-90 -45 0 45 90</ElevationGrid>\n');
fprintf(f,'<AzimuthGrid>-90 0 90</AzimuthGrid>\n');
fprintf(f,'<CouplingPhase>90 90</CouplingPhase>\n');
fprintf(f,'</arrayant></qdant>');
fclose(f);

try
    [~,~,~,~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_qdant_read('test.qdant');
    error('moxunit:exceptionNotRaised', 'Expected an error!');
end

% Not a XML File
f = fopen( 'test.qdant','w' );
fprintf(f,'bla bla bla\n');
fclose(f);

try
    [~,~,~,~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_qdant_read('test.qdant');
    error('moxunit:exceptionNotRaised', 'Expected an error!');
end

% Not a QDANT file (KML instead)
f = fopen( 'test.qdant','w' );
fprintf(f,'<?xml version="1.0" encoding="UTF-8"?>\n');
fprintf(f,'<kml xmlns="http://www.opengis.net/kml/2.2">\n');
fprintf(f,'  <Placemark>\n');
fprintf(f,'    <name>A simple placemark on the ground</name>\n');
fprintf(f,'    <Point>\n');
fprintf(f,'		<coordinates>8.542952335953721,47.36685263064198,0</coordinates>\n');
fprintf(f,'    </Point>\n');
fprintf(f,'  </Placemark>\n');
fprintf(f,'</kml>\n');
fclose(f);

try
    [~,~,~,~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_qdant_read('test.qdant');
    error('moxunit:exceptionNotRaised', 'Expected an error!');
end

delete('test.qdant');


