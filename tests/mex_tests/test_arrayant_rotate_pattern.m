function test_arrayant_rotate_pattern

ant = quadriga_lib.arrayant_generate('custom',[],[],5,20,0);

out = quadriga_lib.arrayant_rotate_pattern(ant, 0, 0, 90);

[~,ii] = max(out.e_theta_re(:));
[s0,s1] = ind2sub(size(out.e_theta_re),ii);
assertEqual(s0,91);
assertEqual(s1,271);

out = quadriga_lib.arrayant_rotate_pattern(ant, 0, -45, 0);

[~,ii] = max(out.e_theta_re(:));
[s0,s1] = ind2sub(size(out.e_theta_re),ii);
assertEqual(s0,136);
assertEqual(s1,181);

v = ant.e_theta_re(91,181);
out = quadriga_lib.arrayant_rotate_pattern(ant, 45, 0, 0);

assertElementsAlmostEqual( out.e_theta_re(91,181) * sqrt(2), v, 'absolute', 1e-14 );
assertElementsAlmostEqual( out.e_phi_re(91,181) * sqrt(2), v, 'absolute', 1e-14 );

out = quadriga_lib.arrayant_rotate_pattern(ant, 180, 180, 180);

assertElementsAlmostEqual( ant.e_theta_re, out.e_theta_re, 'absolute', 1e-13 );

out = quadriga_lib.arrayant_rotate_pattern(ant, 90, 0, 0, 1);

assertElementsAlmostEqual( out.e_theta_re(91,181) , v, 'absolute', 1e-14 );
assertElementsAlmostEqual( ant.e_theta_re(92,181) , out.e_theta_re(91,182), 'absolute', 1e-14 );

out = quadriga_lib.arrayant_rotate_pattern(ant, 90, 0, 0, 2);

assertElementsAlmostEqual( (out.e_theta_re.^2 + out.e_phi_re.^2 - ant.e_theta_re.^2 - ant.e_phi_re.^2) , zeros(181,361), 'absolute', 1e-13 );
assertElementsAlmostEqual( out.e_theta_re(91,181), 0, 'absolute', 1e-14 );
assertElementsAlmostEqual( out.e_phi_re(91,181), v, 'absolute', 1e-14 );

res = 0.05;

azimuth_grid =(-2*res:res:2*res)*pi/180;
elevation_grid =(-res:res:res)*pi/180;
e_theta_re = zeros(numel(elevation_grid),numel(azimuth_grid),2);
e_theta_re(2,2:end-1,1) = 1:numel(azimuth_grid)-2;
e_theta_re(:,:,2) = -e_theta_re(:,:,1);
zr = zeros(numel(elevation_grid),numel(azimuth_grid),2);
element_pos = rand(3,2);
usage = 0;


x = 90;
y = 10;
z = 113;

x = round(x/res)*res;
y = round(y/res)*res;
z = round(z/res)*res;

R = quadriga_lib.calc_rotation_matrix([x;y;z]*pi/180);

[e_theta_re_r, e_theta_im_r, e_phi_re_r, e_phi_im_r, azimuth_grid_r, elevation_grid_r, element_pos_r, ~, ~, ~, ~] ...
    = quadriga_lib.arrayant_rotate_pattern([], x, y, z, usage, [], e_theta_re, zr, zr, zr, azimuth_grid, elevation_grid, element_pos);

assertElementsAlmostEqual( diff( azimuth_grid_r*180/pi ) , ones(1,numel(azimuth_grid_r)-1)*0.05, 'absolute', 1e-12 );
assertElementsAlmostEqual( diff( elevation_grid_r*180/pi ) , ones(1,numel(elevation_grid_r)-1)*0.05, 'absolute', 1e-12 );
assertElementsAlmostEqual( element_pos_r , reshape(R,3,3) * element_pos, 'absolute', 1e-12 );

[~,ii] = min(abs(azimuth_grid_r*180/pi - z));
[~,jj] = min(abs(elevation_grid_r*180/pi + y));

assertElementsAlmostEqual( azimuth_grid_r(ii)*180/pi , z, 'absolute', 1e-12 );
assertElementsAlmostEqual( elevation_grid_r(jj)*180/pi , -y, 'absolute', 1e-12 );

assertTrue( all(abs( e_theta_re_r(:,1) ) < 1e-12) );
assertTrue( all(abs( e_theta_re_r(:,end) ) < 1e-12) );
assertTrue( all(abs( e_theta_re_r(1,:) ) < 1e-12) );
assertTrue( all(abs( e_theta_re_r(end,:) ) < 1e-12) );

assertTrue( all(abs( e_theta_re_r(:,jj) ) < 1e-12) );

assertTrue( all(abs( e_phi_re_r(:,1) ) < 1e-12) );
assertTrue( all(abs( e_phi_re_r(:,end) ) < 1e-12) );
assertTrue( all(abs( e_phi_re_r(1,:) ) < 1e-12) );
assertTrue( all(abs( e_phi_re_r(end,:) ) < 1e-12) );

assertTrue( all(abs( e_theta_im_r(:) ) < 1e-12) );
assertTrue( all(abs( e_phi_im_r(:) ) < 1e-12) );

[e_theta_re_q, e_theta_im_q, e_phi_re_q, e_phi_im_q, azimuth_grid_q, elevation_grid_q, element_pos_q, ~, ~, ~, ~] ...
    = quadriga_lib.arrayant_rotate_pattern([], 0, 0, -z, usage, [], e_theta_re_r, e_theta_im_r, e_phi_re_r, e_phi_im_r, azimuth_grid_r, elevation_grid_r, element_pos_r);

[e_theta_re_q, e_theta_im_q, e_phi_re_q, e_phi_im_q, azimuth_grid_q, elevation_grid_q, element_pos_q, ~, ~, ~, ~] ...
    = quadriga_lib.arrayant_rotate_pattern([], 0, -y, 0, usage, [], e_theta_re_q, e_theta_im_q, e_phi_re_q, e_phi_im_q, azimuth_grid_q, elevation_grid_q, element_pos_q);

[~, ~, ~, ~, ~, ~, element_pos_q, ~, ~, ~, ~] ...
    = quadriga_lib.arrayant_rotate_pattern([], -x, 0, 0, usage, [], e_theta_re_q, e_theta_im_q, e_phi_re_q, e_phi_im_q, azimuth_grid_q, elevation_grid_q, element_pos_q);

assertElementsAlmostEqual( element_pos, element_pos_q, 'absolute', 1e-12 );

% Single element rotation
try
    out = quadriga_lib.arrayant_rotate_pattern(ant, 0, 0, 90, 1, 3);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input parameter ''element'' out of bound.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

ant = quadriga_lib.arrayant_copy_element(ant,1,[2,3]);
ant = quadriga_lib.arrayant_copy_element(ant,[1,2],[2,3]);
out = quadriga_lib.arrayant_rotate_pattern(ant, 0, 0, 90, [], [2,3]);

tmp = out.e_theta_re(:,:,1);
[~,ii] = max(tmp(:));
[s0,s1] = ind2sub(size(tmp),ii);
assertEqual(s0,91);
assertEqual(s1,181);

tmp = out.e_theta_re(:,:,2);
[~,ii] = max(tmp(:));
[s0,s1] = ind2sub(size(tmp),ii);
assertEqual(s0,91);
assertEqual(s1,271);

tmp = out.e_theta_re(:,:,3);
[~,ii] = max(tmp(:));
[s0,s1] = ind2sub(size(tmp),ii);
assertEqual(s0,91);
assertEqual(s1,271);

% Errors
try
    [~, ~] = quadriga_lib.arrayant_rotate_pattern( e_theta_re );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input must be a struct.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [~, ~] = quadriga_lib.arrayant_rotate_pattern( ant );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end

