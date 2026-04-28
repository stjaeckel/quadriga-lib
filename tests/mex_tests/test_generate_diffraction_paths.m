function test_generate_diffraction_paths

orig = [0, 0, 10];
dest = [100, 0, 10];
fc = 2.4e9;

% Verify output shapes and weight normalization for each lod level
lod_table = [1, 7, 3; 2, 19, 3; 3, 37, 4; 4, 61, 5; 5, 1, 2; 6, 2, 2];

for k = 1:size(lod_table, 1)
    lod = lod_table(k, 1);
    n_path_exp = lod_table(k, 2);
    n_seg_exp = lod_table(k, 3);

    [rays, weights] = quadriga_lib.generate_diffraction_paths(orig, dest, fc, lod);

    assertTrue( isa(rays, 'double') );
    assertTrue( isa(weights, 'double') );

    % Pad size() output to expected rank (MATLAB strips trailing singletons)
    sz_rays = size(rays);
    sz_weights = size(weights);
    sz_rays = [sz_rays, ones(1, 4 - numel(sz_rays))];
    sz_weights = [sz_weights, ones(1, 3 - numel(sz_weights))];

    assertTrue( isequal( sz_rays, [1, n_path_exp, n_seg_exp - 1, 3] ) );
    assertTrue( isequal( sz_weights, [1, n_path_exp, n_seg_exp] ) );

    % Weight normalization: sum(prod(weights,3),2) = 1 per position
    norm_check = sum(prod(weights, 3), 2);
    assertTrue( all( abs(norm_check(:) - 1) < 1e-6 ) );
end

% Multiple positions
n_pos = 5;
orig_m = repmat([0, 0, 10], n_pos, 1);
dest_m = repmat([100, 0, 10], n_pos, 1);
dest_m(:, 1) = (50 : 50/(n_pos-1) : 100)';

[rays, weights] = quadriga_lib.generate_diffraction_paths(orig_m, dest_m, fc, 2);
assertTrue( isequal( size(rays), [n_pos, 19, 2, 3] ) );
assertTrue( isequal( size(weights), [n_pos, 19, 3] ) );

norm_check = sum(prod(weights, 3), 2);
assertTrue( all( abs(norm_check(:) - 1) < 1e-6 ) );

% Single output (rays only) — exercises p_weight = nullptr path
rays_only = quadriga_lib.generate_diffraction_paths(orig, dest, fc, 1);
assertTrue( isequal( size(rays_only), [1, 7, 2, 3] ) );

% --- Error tests ---

try % 3 inputs
    quadriga_lib.generate_diffraction_paths(orig, dest, fc);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 5 inputs
    quadriga_lib.generate_diffraction_paths(orig, dest, fc, 1, 99);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 3 outputs
    [~,~,~] = quadriga_lib.generate_diffraction_paths(orig, dest, fc, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % invalid lod (0, below range)
    quadriga_lib.generate_diffraction_paths(orig, dest, fc, 0);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'lod';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % invalid lod (7, above range)
    quadriga_lib.generate_diffraction_paths(orig, dest, fc, 7);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'lod';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % orig wrong column count
    quadriga_lib.generate_diffraction_paths(orig(:, 1:2), dest, fc, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'orig';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % dest wrong column count
    quadriga_lib.generate_diffraction_paths(orig, dest(:, 1:2), fc, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''dest'' must have 3 columns containing the x,y,z coordinates.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % orig/dest row count mismatch
    quadriga_lib.generate_diffraction_paths([orig; orig], dest, fc, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Inputs ''orig'' and ''dest'' must have the same number of rows.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end
