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
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:no_output');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:no_output',expt.identifier);
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate([],e,e,e,az_grid,el_grid,az,el);
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:import_error');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:import_error',expt.identifier);
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,[],e,e,az_grid,el_grid,az,el);
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:import_error');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:import_error',expt.identifier);
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,[],el_grid,az,el);
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:import_error');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:import_error',expt.identifier);
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,[],el);
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:import_error');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:import_error',expt.identifier);
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,[]);
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:import_error');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:import_error',expt.identifier);
end

% All double
[~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el);

% Mixed types
e = single(e);
try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el);
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:wrong_type');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:wrong_type',expt.identifier);
end

az_grid = single(az_grid);
el_grid = single(el_grid);
az = single(az);
el = single(el);

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e(1,:,:),e,e,az_grid,el_grid,az,el);
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:import_error');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:import_error',expt.identifier);
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e(:,1,:),e,az_grid,el_grid,az,el);
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:import_error');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:import_error',expt.identifier);
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e(:,:,1),az_grid,el_grid,az,el);
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:import_error');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:import_error',expt.identifier);
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid(1:2),el_grid,az,el);
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:import_error');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:import_error',expt.identifier);
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid(1:2),az,el);
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:import_error');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:import_error',expt.identifier);
end

try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el(1:2));
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:size_mismatch');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:size_mismatch',expt.identifier);
end

% Should be error-free
[~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el);
[~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,[]);

% Test azimuth grid range
try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,2*az_grid,el_grid,az,el);
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:import_error');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:import_error',expt.identifier);
end

az(1) = pi;
[~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el);

% Test elevation grid range (-pi/ to pi/2)
try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,2*el_grid,az,el);
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:import_error');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:import_error',expt.identifier);
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
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:out_of_bound');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:out_of_bound',expt.identifier);
end

% i_element cannot be -1
try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,-1);
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:out_of_bound');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:out_of_bound',expt.identifier);
end

% i_element cannot exceed number of elements
try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,[1,4]);
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:out_of_bound');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:out_of_bound',expt.identifier);
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
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:size_mismatch');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:size_mismatch',expt.identifier);
end

% orientation must be double or single
try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,[],int32(or));
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:wrong_type');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:wrong_type',expt.identifier);
end

% Optional per-element angles 
try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az([1,1],:),el,[1,2]);
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:size_mismatch');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:size_mismatch',expt.identifier);
end
try
    [~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az([1,1,1],:),el([1,1,1],:),[1,2]);
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:size_mismatch');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:size_mismatch',expt.identifier);
end
[~, ~, ~, ~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az([1,1],:),el([1,1],:),[1,1]);

% 11 inputs should throw error
try
    [~,~,~,~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,[],[],[],[]);
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:no_input');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:no_input',expt.identifier);
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

% 8 outputs should throw error
try
    [~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_interpolate(e,e,e,e,az_grid,el_grid,az,el,[],[]);
    error_exception_not_thrown('quadriga_lib:arrayant_interpolate:no_output');
catch expt
    error_if_wrong_id_thrown('quadriga_lib:arrayant_interpolate:no_output',expt.identifier);
end



% ---------------- HELPER FUNCTIONS ------------------
function error_exception_not_thrown(error_id)
error('moxunit:exceptionNotRaised', 'Exception ''%s'' not thrown', error_id);

function error_if_wrong_id_thrown(expected_error_id, thrown_error_id)
if ~strcmp(thrown_error_id, expected_error_id)
    error('moxunit:wrongExceptionRaised',...
        'Exception raised with id ''%s'' expected id ''%s''',...
        thrown_error_id,expected_error_id);
end
