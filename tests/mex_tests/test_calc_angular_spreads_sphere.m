function test_calc_angular_spreads_sphere()
% MOxUnit tests for quadriga_lib.calc_angular_spreads_sphere

% --- Single path returns zero spread ---
az = 0.5;
el = 0.3;
powers = 1.0;
[as, es, orient] = quadriga_lib.calc_angular_spreads_sphere(az, el, powers);
assertElementsAlmostEqual(as, 0.0, 'absolute', 1e-10);
assertElementsAlmostEqual(es, 0.0, 'absolute', 1e-10);
assertElementsAlmostEqual(orient(3,1), 0.5, 'absolute', 1e-10);   % heading ~ azimuth
assertElementsAlmostEqual(orient(2,1), 0.3, 'absolute', 1e-10);   % tilt ~ elevation

% --- Two symmetric azimuth paths ---
az = [0.1; -0.1];
el = [0.0; 0.0];
powers = [1.0; 1.0];
[as, es, orient] = quadriga_lib.calc_angular_spreads_sphere(az, el, powers);
assertElementsAlmostEqual(as, 0.1, 'absolute', 1e-6);
assertElementsAlmostEqual(es, 0.0, 'absolute', 1e-6);
assertElementsAlmostEqual(orient(3,1), 0.0, 'absolute', 1e-6);

% --- Two symmetric elevation paths ---
az = [0.0; 0.0];
el = [0.2; -0.2];
powers = [1.0; 1.0];
[as, es, orient, phi, theta] = quadriga_lib.calc_angular_spreads_sphere(az, el, powers);
assertEqual(size(phi), [2, 1]);
assertEqual(size(theta), [2, 1]);
% After bank rotation, AS >= ES
assertTrue(as >= es - 1e-6);
% Total angular extent preserved
total = sqrt(as^2 + es^2);
assertElementsAlmostEqual(total, 0.2, 'absolute', 1e-4);

% --- Pole paths (near zenith) ---
az = [0.0; pi/2; pi; -pi/2];
el = [1.4; 1.4; 1.4; 1.4];
powers = [1.0; 1.0; 1.0; 1.0];
[as, es] = quadriga_lib.calc_angular_spreads_sphere(az, el, powers);
assertTrue(as < 0.5);
assertTrue(es < 0.5);

% --- Multiple CIRs with same path count (2 CIRs, 3 paths each) ---
az = [0.1, 0.2; -0.1, -0.2; 0.0, 0.0];
el = zeros(3, 2);
powers = ones(3, 2);
[as, es, orient, phi, theta] = quadriga_lib.calc_angular_spreads_sphere(az, el, powers);
assertEqual(size(as), [2, 1]);
assertEqual(size(orient), [3, 2]);
assertEqual(size(phi), [3, 2]);
assertTrue(as(2) > as(1));

% --- No bank angle calculation ---
az = [0.0; 0.0];
el = [0.2; -0.2];
powers = [1.0; 1.0];
[as_bank, ~, orient_bank] = quadriga_lib.calc_angular_spreads_sphere(az, el, powers, false, true);
[as_nobank, ~, orient_nobank] = quadriga_lib.calc_angular_spreads_sphere(az, el, powers, false, false);
assertElementsAlmostEqual(orient_nobank(1,1), 0.0, 'absolute', 1e-10);
assertTrue(as_bank >= as_nobank - 1e-6);

% --- Disable wrapping ---
az = [0.1; -0.1; 0.05];
el = [0.2; -0.2; 0.0];
powers = [1.0; 1.0; 1.0];
[as_raw, es_raw, orient_raw, phi_raw, theta_raw] = quadriga_lib.calc_angular_spreads_sphere(az, el, powers, true);
assertElementsAlmostEqual(orient_raw(:,1), [0; 0; 0], 'absolute', 1e-10);
assertElementsAlmostEqual(phi_raw, az, 'absolute', 1e-14);
assertElementsAlmostEqual(theta_raw, el, 'absolute', 1e-14);

% --- Quantization groups nearby paths ---
az = [0.0; 0.01];
el = [0.0; 0.0];
powers = [1.0; 1.0];
[as_raw] = quadriga_lib.calc_angular_spreads_sphere(az, el, powers, false, true, 0.0);
[as_quant] = quadriga_lib.calc_angular_spreads_sphere(az, el, powers, false, true, 3.0);
assertTrue(as_quant <= as_raw + 1e-8);

% --- Wrap-around at +/- pi ---
az = [3.0; -3.0];
el = [0.0; 0.0];
powers = [1.0; 1.0];
[as] = quadriga_lib.calc_angular_spreads_sphere(az, el, powers);
gap = 2 * (pi - 3.0);
assertTrue(as < gap);
assertTrue(as > 0);

% --- Output shapes for 1 CIR, 5 paths ---
az = [0.3; -0.2; 0.1; -0.4; 0.0];
el = [0.1; -0.1; 0.05; -0.05; 0.0];
powers = ones(5, 1);
[as, es, orient, phi, theta] = quadriga_lib.calc_angular_spreads_sphere(az, el, powers);
assertEqual(size(as), [1, 1]);
assertEqual(size(es), [1, 1]);
assertEqual(size(orient), [3, 1]);
assertEqual(size(phi), [5, 1]);
assertEqual(size(theta), [5, 1]);

% --- Rotated angles in valid range ---
assertTrue(all(phi >= -pi - 0.01));
assertTrue(all(phi <= pi + 0.01));
assertTrue(all(theta >= -pi/2 - 0.01));
assertTrue(all(theta <= pi/2 + 0.01));

% --- Error: too few arguments ---
try
    quadriga_lib.calc_angular_spreads_sphere(az, el);
    error('Expected an error');
catch
    % Expected
end

end
