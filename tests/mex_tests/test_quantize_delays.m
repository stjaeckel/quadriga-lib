function test_quantize_delays()
% MOxUnit tests for quadriga_lib.quantize_delays

% --- Input validation tests ---
cre = ones(2, 2, 3, 1);
cim = zeros(2, 2, 3, 1);
dl = rand(2, 2, 3, 1) * 100e-9;

% Invalid tap_spacing
try
    quadriga_lib.quantize_delays(cre, cim, dl, 0.0);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'tap_spacing';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "', ME.message, '"']);
    end
end

% Invalid fix_taps
try
    quadriga_lib.quantize_delays(cre, cim, dl, 5e-9, 48, 1.0, -1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'fix_taps';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "', ME.message, '"']);
    end
end

% --- Single path at exact tap boundary ---
cre = zeros(1, 1, 1, 1);
cim = zeros(1, 1, 1, 1);
dl = zeros(1, 1, 1, 1);
cre(1,1,1,1) = 2.0;
cim(1,1,1,1) = 3.0;
dl(1,1,1,1) = 15e-9;

[cre_q, cim_q, dl_q] = quadriga_lib.quantize_delays(cre, cim, dl, 5e-9);

found = false;
for k = 1:size(cre_q, 3)
    if abs(cre_q(1,1,k,1)) > 1e-10
        assertElementsAlmostEqual(cre_q(1,1,k,1), 2.0, 'absolute', 1e-12);
        assertElementsAlmostEqual(cim_q(1,1,k,1), 3.0, 'absolute', 1e-12);
        assertElementsAlmostEqual(dl_q(1,1,k,1), 15e-9, 'absolute', 1e-20);
        found = true;
    end
end
assertEqual(found, true);

% --- Half-tap offset with linear exponent ---
cre = zeros(1, 1, 1, 1);
cim = zeros(1, 1, 1, 1);
dl = zeros(1, 1, 1, 1);
cre(1,1,1,1) = 4.0;
dl(1,1,1,1) = 12.5e-9;

[cre_q, ~, ~] = quadriga_lib.quantize_delays(cre, cim, dl, 5e-9, 48, 1.0);

n_nonzero = 0;
sum_re = 0;
for k = 1:size(cre_q, 3)
    v = cre_q(1,1,k,1);
    if abs(v) > 1e-10
        assertElementsAlmostEqual(v, 2.0, 'absolute', 1e-12);
        sum_re = sum_re + v;
        n_nonzero = n_nonzero + 1;
    end
end
assertEqual(n_nonzero, 2);
assertElementsAlmostEqual(sum_re, 4.0, 'absolute', 1e-12);

% --- Half-tap offset with sqrt exponent ---
cre = zeros(1, 1, 1, 1);
cim = zeros(1, 1, 1, 1);
dl = zeros(1, 1, 1, 1);
cre(1,1,1,1) = 4.0;
dl(1,1,1,1) = 12.5e-9;

[cre_q, ~, ~] = quadriga_lib.quantize_delays(cre, cim, dl, 5e-9, 48, 0.5);

expected_coeff = sqrt(0.5) * 4.0;
sum_power = 0;
n_nonzero = 0;
for k = 1:size(cre_q, 3)
    v = cre_q(1,1,k,1);
    if abs(v) > 1e-10
        assertElementsAlmostEqual(v, expected_coeff, 'absolute', 1e-12);
        sum_power = sum_power + v * v;
        n_nonzero = n_nonzero + 1;
    end
end
assertEqual(n_nonzero, 2);
assertElementsAlmostEqual(sum_power, 16.0, 'absolute', 1e-10);

% --- Already quantized input ---
cre = zeros(1, 1, 3, 1);
cim = zeros(1, 1, 3, 1);
dl = zeros(1, 1, 3, 1);
cre(1,1,1,1) = 1.0; cim(1,1,1,1) = 0.5; dl(1,1,1,1) = 0.0;
cre(1,1,2,1) = 2.0; cim(1,1,2,1) = 1.0; dl(1,1,2,1) = 5e-9;
cre(1,1,3,1) = 0.5; cim(1,1,3,1) = 0.3; dl(1,1,3,1) = 20e-9;

[cre_q, cim_q, dl_q] = quadriga_lib.quantize_delays(cre, cim, dl, 5e-9);

for k = 1:size(cre_q, 3)
    d = dl_q(1,1,k,1);
    re = cre_q(1,1,k,1);
    im = cim_q(1,1,k,1);
    if abs(d) < 1e-20
        assertElementsAlmostEqual(re, 1.0, 'absolute', 1e-12);
        assertElementsAlmostEqual(im, 0.5, 'absolute', 1e-12);
    elseif abs(d - 5e-9) < 1e-20
        assertElementsAlmostEqual(re, 2.0, 'absolute', 1e-12);
        assertElementsAlmostEqual(im, 1.0, 'absolute', 1e-12);
    elseif abs(d - 20e-9) < 1e-20
        assertElementsAlmostEqual(re, 0.5, 'absolute', 1e-12);
        assertElementsAlmostEqual(im, 0.3, 'absolute', 1e-12);
    end
end

% --- max_no_taps limits output ---
cre = zeros(1, 1, 5, 1);
cim = zeros(1, 1, 5, 1);
dl = zeros(1, 1, 5, 1);
cre(1,1,1,1) = 5.0; dl(1,1,1,1) = 0.0;
cre(1,1,2,1) = 1.0; dl(1,1,2,1) = 5e-9;
cre(1,1,3,1) = 3.0; dl(1,1,3,1) = 10e-9;
cre(1,1,4,1) = 0.5; dl(1,1,4,1) = 15e-9;
cre(1,1,5,1) = 4.0; dl(1,1,5,1) = 20e-9;

[cre_q, ~, ~] = quadriga_lib.quantize_delays(cre, cim, dl, 5e-9, 3);

n_nonzero = 0;
for k = 1:size(cre_q, 3)
    if abs(cre_q(1,1,k,1)) > 1e-10
        n_nonzero = n_nonzero + 1;
    end
end
assertEqual(n_nonzero, 3);
assertEqual(double(size(cre_q, 3) <= 3), 1.0);

% --- Shared delays with fix_taps=2 ---
cre4 = randn(2, 2, 3, 2);
cim4 = randn(2, 2, 3, 2);
dl4 = zeros(1, 1, 3, 2);
dl4(1,1,1,:) = 0.0;
dl4(1,1,2,:) = 12.5e-9;
dl4(1,1,3,:) = 30e-9;

[~, ~, dl_q] = quadriga_lib.quantize_delays(cre4, cim4, dl4, 5e-9, 48, 1.0, 2);

assertEqual(size(dl_q, 1), 1);
assertEqual(size(dl_q, 2), 1);

% --- Shared delays with fix_taps=0 produces per-antenna output ---
cre4 = randn(2, 2, 2, 1);
cim4 = randn(2, 2, 2, 1);
dl4 = zeros(1, 1, 2, 1);
dl4(1,1,1,1) = 0.0;
dl4(1,1,2,1) = 17.5e-9;

[~, ~, dl_q] = quadriga_lib.quantize_delays(cre4, cim4, dl4, 5e-9, 48, 1.0, 0);

assertEqual(size(dl_q, 1), 2);
assertEqual(size(dl_q, 2), 2);

% --- Multiple snapshots with fix_taps=1 ---
cre4 = ones(2, 1, 2, 2);
cim4 = zeros(2, 1, 2, 2);
dl4 = zeros(2, 1, 2, 2);
dl4(1,1,1,:) = 0.0;
dl4(1,1,2,:) = 12.5e-9;
dl4(2,1,1,:) = 5e-9;
dl4(2,1,2,:) = 27.5e-9;

[~, ~, dl_q] = quadriga_lib.quantize_delays(cre4, cim4, dl4, 5e-9, 48, 1.0, 1);

assertEqual(size(dl_q, 4), 2);
n_taps = size(dl_q, 3);
for k = 1:n_taps
    assertElementsAlmostEqual(dl_q(1,1,k,1), dl_q(1,1,k,2), 'absolute', 1e-20);
end

% --- Paths combining at same tap ---
cre = zeros(1, 1, 2, 1);
cim = zeros(1, 1, 2, 1);
dl = zeros(1, 1, 2, 1);
cre(1,1,1,1) = 1.0; dl(1,1,1,1) = 10e-9;
cre(1,1,2,1) = 2.0; dl(1,1,2,1) = 10e-9;

[cre_q, ~, dl_q] = quadriga_lib.quantize_delays(cre, cim, dl, 5e-9);

found = false;
for k = 1:size(cre_q, 3)
    if abs(dl_q(1,1,k,1) - 10e-9) < 1e-20
        assertElementsAlmostEqual(cre_q(1,1,k,1), 3.0, 'absolute', 1e-12);
        found = true;
    end
end
assertEqual(found, true);

% --- Zero delay ---
cre = zeros(1, 1, 1, 1);
cim = zeros(1, 1, 1, 1);
dl = zeros(1, 1, 1, 1);
cre(1,1,1,1) = 1.0;
cim(1,1,1,1) = -1.0;

[cre_q, cim_q, dl_q] = quadriga_lib.quantize_delays(cre, cim, dl, 5e-9);

assertElementsAlmostEqual(cre_q(1,1,1,1), 1.0, 'absolute', 1e-12);
assertElementsAlmostEqual(cim_q(1,1,1,1), -1.0, 'absolute', 1e-12);
assertElementsAlmostEqual(dl_q(1,1,1,1), 0.0, 'absolute', 1e-20);

end
