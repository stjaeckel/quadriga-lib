function test_calc_angular_spreads_sphere()
% MOxUnit tests for quadriga_lib.calc_angular_spreads_sphere

% --- Single path returns zero spread ---
az = 0.5;
el = 0.3;
pw = 1.0;
[as, es, orient] = quadriga_lib.calc_angular_spreads_sphere(az, el, pw);
assertElementsAlmostEqual(as, 0.0, 'absolute', 1e-10);
assertElementsAlmostEqual(es, 0.0, 'absolute', 1e-10);
assertElementsAlmostEqual(orient(3,1), 0.5, 'absolute', 1e-10);   % heading ~ azimuth
assertElementsAlmostEqual(orient(2,1), 0.3, 'absolute', 1e-10);   % tilt ~ elevation

% --- Two symmetric azimuth paths ---
az = [0.1, -0.1];
el = [0.0, 0.0];
pw = [1.0, 1.0];
[as, es, orient] = quadriga_lib.calc_angular_spreads_sphere(az, el, pw);
assertElementsAlmostEqual(as, 0.1, 'absolute', 1e-6);
assertElementsAlmostEqual(es, 0.0, 'absolute', 1e-6);
assertElementsAlmostEqual(orient(3,1), 0.0, 'absolute', 1e-6);

% --- Two symmetric elevation paths ---
az = [0.0, 0.0];
el = [0.2, -0.2];
pw = [1.0, 1.0];
[as, es, orient, phi, theta] = quadriga_lib.calc_angular_spreads_sphere(az, el, pw);
assertEqual(size(phi), [1, 2]);
assertEqual(size(theta), [1, 2]);
% After bank rotation, AS >= ES
assertTrue(as >= es - 1e-6);
% Total angular extent preserved
total = sqrt(as^2 + es^2);
assertElementsAlmostEqual(total, 0.2, 'absolute', 1e-4);

% --- Pole paths (near zenith) ---
az = [0.0, pi/2, pi, -pi/2];
el = [1.4, 1.4, 1.4, 1.4];
pw = [1.0, 1.0, 1.0, 1.0];
[as, es] = quadriga_lib.calc_angular_spreads_sphere(az, el, pw);
assertTrue(as < 0.5);
assertTrue(es < 0.5);

% --- Power broadcasting (single pow row, multiple angle sets) ---
az = [0.1, -0.1, 0.0; 0.2, -0.2, 0.0];
el = zeros(2, 3);
pw = [1.0, 1.0, 1.0];  % 1 row
[as, es] = quadriga_lib.calc_angular_spreads_sphere(az, el, pw);
assertEqual(size(as), [2, 1]);
assertTrue(as(2) > as(1));

% --- No bank angle calculation ---
az = [0.0, 0.0];
el = [0.2, -0.2];
pw = [1.0, 1.0];
[as_bank, ~, orient_bank] = quadriga_lib.calc_angular_spreads_sphere(az, el, pw, true);
[as_nobank, ~, orient_nobank] = quadriga_lib.calc_angular_spreads_sphere(az, el, pw, false);
assertElementsAlmostEqual(orient_nobank(1,1), 0.0, 'absolute', 1e-10);
assertTrue(as_bank >= as_nobank - 1e-6);

% --- Quantization groups nearby paths ---
az = [0.0, 0.01];
el = [0.0, 0.0];
pw = [1.0, 1.0];
[as_raw] = quadriga_lib.calc_angular_spreads_sphere(az, el, pw, true, 0.0);
[as_quant] = quadriga_lib.calc_angular_spreads_sphere(az, el, pw, true, 3.0);
assertTrue(as_quant <= as_raw + 1e-8);

% --- Wrap-around at +/- pi ---
az = [3.0, -3.0];
el = [0.0, 0.0];
pw = [1.0, 1.0];
[as] = quadriga_lib.calc_angular_spreads_sphere(az, el, pw);
gap = 2 * (pi - 3.0);
assertTrue(as < gap);
assertTrue(as > 0);

% --- Multiple angle sets ---
az = [0.01, -0.01, 0.005, -0.005; ...
      0.1, -0.1, 0.05, -0.05; ...
      1.0, -1.0, 0.5, -0.5];
el = zeros(3, 4);
pw = ones(3, 4);
[as, es, orient, phi, theta] = quadriga_lib.calc_angular_spreads_sphere(az, el, pw);
assertEqual(size(as), [3, 1]);
assertEqual(size(orient), [3, 3]);
assertEqual(size(phi), [3, 4]);
assertTrue(as(1) < as(2));
assertTrue(as(2) < as(3));

% --- Output shapes for 1 angle set, 5 paths ---
az = [0.3, -0.2, 0.1, -0.4, 0.0];
el = [0.1, -0.1, 0.05, -0.05, 0.0];
pw = ones(1, 5);
[as, es, orient, phi, theta] = quadriga_lib.calc_angular_spreads_sphere(az, el, pw);
assertEqual(size(as), [1, 1]);
assertEqual(size(es), [1, 1]);
assertEqual(size(orient), [3, 1]);
assertEqual(size(phi), [1, 5]);
assertEqual(size(theta), [1, 5]);

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
