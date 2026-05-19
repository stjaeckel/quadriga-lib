function test_arrayant_rotate_pattern

% --- Single-frequency tests ---

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
    expectedErrorMessage = 'First input must be an arrayant struct/struct array';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [~,~,~,~,~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_rotate_pattern( ant );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% --- Multi-frequency (struct-array) tests ---

% Build a 2-entry frequency-dependent struct array
mf_ant = quadriga_lib.arrayant_generate('custom',[],[],5,20,0);
mf_ant.center_freq = 1e9;  % ensure field exists for struct-array assembly

clear mf_multi
mf_multi(1) = mf_ant;
mf_multi(2) = mf_ant;
mf_multi(1).center_freq = 3.5e9;
mf_multi(2).center_freq = 28e9;

% --- Basic dispatch: struct array in -> struct array out ---
out_mf = quadriga_lib.arrayant_rotate_pattern(mf_multi, 0, 0, 90);
assertEqual(numel(out_mf), 2);

for k = 1:2
    [~,ii] = max(out_mf(k).e_theta_re(:));
    [s0,s1] = ind2sub(size(out_mf(k).e_theta_re), ii);
    assertEqual(s0, 91);
    assertEqual(s1, 271);
end

% Per-entry center_freq preserved
assertElementsAlmostEqual(out_mf(1).center_freq, 3.5e9, 'absolute', 1e-3);
assertElementsAlmostEqual(out_mf(2).center_freq, 28e9,  'absolute', 1e-3);

% Input struct array not mutated (copy=true in wrapper)
assertElementsAlmostEqual(mf_multi(1).e_theta_re, mf_ant.e_theta_re, 'absolute', 1e-14);
assertElementsAlmostEqual(mf_multi(2).e_theta_re, mf_ant.e_theta_re, 'absolute', 1e-14);

% Grid not adjusted in multi-freq mode (always no-grid-adj internally)
assertElementsAlmostEqual(out_mf(1).azimuth_grid,   mf_ant.azimuth_grid,   'absolute', 1e-14);
assertElementsAlmostEqual(out_mf(1).elevation_grid, mf_ant.elevation_grid, 'absolute', 1e-14);
assertElementsAlmostEqual(out_mf(2).azimuth_grid,   mf_ant.azimuth_grid,   'absolute', 1e-14);
assertElementsAlmostEqual(out_mf(2).elevation_grid, mf_ant.elevation_grid, 'absolute', 1e-14);

% --- Equivalence with single-freq no-grid-adj modes ---
% multi usage=0 maps internally to single-freq usage=3 (both, no grid adj)
% multi usage=1 maps internally to single-freq usage=4 (pattern only, no grid adj)
% multi usage=2 stays at single-freq usage=2 (pol only, no grid adj)
ref_0 = quadriga_lib.arrayant_rotate_pattern(mf_ant, 30, 45, 60, 3);
ref_1 = quadriga_lib.arrayant_rotate_pattern(mf_ant, 30, 45, 60, 4);
ref_2 = quadriga_lib.arrayant_rotate_pattern(mf_ant, 30, 45, 60, 2);

out_0 = quadriga_lib.arrayant_rotate_pattern(mf_multi, 30, 45, 60, 0);
out_1 = quadriga_lib.arrayant_rotate_pattern(mf_multi, 30, 45, 60, 1);
out_2 = quadriga_lib.arrayant_rotate_pattern(mf_multi, 30, 45, 60, 2);

for k = 1:2
    assertElementsAlmostEqual(out_0(k).e_theta_re, ref_0.e_theta_re, 'absolute', 1e-12);
    assertElementsAlmostEqual(out_0(k).e_theta_im, ref_0.e_theta_im, 'absolute', 1e-12);
    assertElementsAlmostEqual(out_0(k).e_phi_re,   ref_0.e_phi_re,   'absolute', 1e-12);
    assertElementsAlmostEqual(out_0(k).e_phi_im,   ref_0.e_phi_im,   'absolute', 1e-12);

    assertElementsAlmostEqual(out_1(k).e_theta_re, ref_1.e_theta_re, 'absolute', 1e-12);
    assertElementsAlmostEqual(out_1(k).e_phi_re,   ref_1.e_phi_re,   'absolute', 1e-12);

    assertElementsAlmostEqual(out_2(k).e_theta_re, ref_2.e_theta_re, 'absolute', 1e-12);
    assertElementsAlmostEqual(out_2(k).e_phi_re,   ref_2.e_phi_re,   'absolute', 1e-12);
end

% --- Element selection in multi-freq mode ---
mf_ant3 = quadriga_lib.arrayant_copy_element(mf_ant, 1, [2,3]);  % 3-element antenna
clear mf_multi3
mf_multi3(1) = mf_ant3;
mf_multi3(2) = mf_ant3;
mf_multi3(1).center_freq = 3.5e9;
mf_multi3(2).center_freq = 28e9;

% Rotate only elements 2 and 3 (element 1 stays put)
out_mf = quadriga_lib.arrayant_rotate_pattern(mf_multi3, 0, 0, 90, 0, [2,3]);

for k = 1:2
    % Element 1 unrotated
    tmp = out_mf(k).e_theta_re(:,:,1);
    [~,ii] = max(tmp(:));
    [s0,s1] = ind2sub(size(tmp), ii);
    assertEqual(s0, 91);
    assertEqual(s1, 181);

    % Elements 2, 3 rotated
    for e = 2:3
        tmp = out_mf(k).e_theta_re(:,:,e);
        [~,ii] = max(tmp(:));
        [s0,s1] = ind2sub(size(tmp), ii);
        assertEqual(s0, 91);
        assertEqual(s1, 271);
    end
end

% --- Error: multi-freq with multiple outputs ---
try
    [~, ~] = quadriga_lib.arrayant_rotate_pattern(mf_multi, 0, 0, 90);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Multi-frequency output supports only struct output';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end


end

