function test_interpolate_exceptions

e = rand( 5,10,3 );
az_grid = linspace(-pi,pi,10);
el_grid = linspace(-pi/2, pi/2, 5);
az = 2*pi*(rand(1,6)-0.5);
el = pi*(rand(1,6)-0.5);

f = @() quadriga_lib.arrayant_interpolate;
assertExceptionThrown( f, 'quadriga_lib:arrayant_interpolate:no_input')

f = @() quadriga_lib.arrayant_interpolate(e);
assertExceptionThrown( f, 'quadriga_lib:arrayant_interpolate:no_input')

f = @() quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el);
assertExceptionThrown( f, 'quadriga_lib:arrayant_interpolate:no_output')

try
    [~, ~, ~] = quadriga_lib.arrayant_interpolate([],e,e,e,az_grid,el_grid,az,el);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate([],e,e,e,az_grid,el_grid,az,el);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,[],e,e,az_grid,el_grid,az,el);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,[],el_grid,az,el);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,[],el);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,[]);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% All double
[~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el);

% Mixed types
e = single(e);
try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

az_grid = single(az_grid);
el_grid = single(el_grid);
az = single(az);
el = single(el);

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e(1,:,:),e,e,az_grid,el_grid,az,el);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e(:,1,:),e,az_grid,el_grid,az,el);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e(:,:,1),az_grid,el_grid,az,el);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid(1:2),el_grid,az,el);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid(1:2),az,el);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el(1:2));
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Should be error-free
[~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el);
[~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,[]);

% Test azimuth grid range
try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,2*az_grid,el_grid,az,el);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

az(1) = pi;
[~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el);

% Test elevation grid range (-pi/ to pi/2)
try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,2*el_grid,az,el);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

el(1) = pi/2;
[~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el);

% i_element must be double or uint32
[Vr, Vi, Hr, Hi] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,uint32([1,2]),[]);

% Rows in output must match elements of i_element
assertEqual(size(Vr),[2,6])
assertEqual(size(Vr),size(Vi))
assertEqual(size(Vr),size(Hr))
assertEqual(size(Vr),size(Hi))

% i_element must be double or uint32
[Vr, Vi, Hr, Hi] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,[1,1,3,3]);

% Rows in output must match elements of i_element
assertEqual(size(Vr),[4,6])
assertEqual(size(Vr),size(Vi))
assertEqual(size(Vr),size(Hr))
assertEqual(size(Vr),size(Hi))

% i_element must be of supported type
[~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,single(3));
[~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,int64(3));
[~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,uint64(3));

% i_element cannot be zero
try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,0);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% i_element cannot be -1
try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,-1);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% i_element cannot exceed number of elements
try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,[1,4]);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end


or = [0,0,1]';

% orientation must be double or single
[Vr, Vi, Hr, Hi] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,[],single(or));

% Rows in output must match number of elements
assertEqual(size(Vr),[3,6])
assertEqual(size(Vr),size(Vi))
assertEqual(size(Vr),size(Hr))
assertEqual(size(Vr),size(Hi))

% orientation must have 3 elements
try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,[],single(1));
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% orientation must be double or single
try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,[],int32(or));
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Optional per-element angles 
try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az([1,1],:),el,[1,2]);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az([1,1,1],:),el([1,1,1],:),[1,2]);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

[~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az([1,1],:),el([1,1],:),[1,1]);

% 11 inputs should throw error
try
    [~,~,~,~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,[],[],[],[]);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% Test projected distance
[~,~,~,~,dist] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el);
assertEqual(size(dist),[3,6])
assertTrue( all( dist(:)==0 ) )

[~,~,~,~,dist] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,[],[],eye(3,'single'));
assertElementsAlmostEqual( sum(abs(dist).^2) ,ones(1,6,'single'), 1e-5)

% Optional output "azimuth_loc"
[~,~,~,~,~,azimuth_loc] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,[],[]);
assertEqual(size(azimuth_loc),[3,6])

% Optional output "azimuth_loc"
[~,~,~,~,~,~,elevation_loc] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,[],[]);
assertEqual(size(elevation_loc),[3,6])

% 9 outputs should throw error
try
    [~,~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,[],[]);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end
