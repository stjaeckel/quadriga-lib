function test_write_png
% MOxUnit test for quadriga_lib.write_png (MEX)

    % --- Test data (matches the Python test) ---
    N = 50;
    max_val = 10.0;
    v = linspace(0.0, max_val, N+1); % mimic endpoint=False
    v(end) = [];
    gradient = repmat(v, 25, 1);
    gradient(1, :) = 0.0;
    gradient(end, :) = 10.0;

    % --- Temp output file ---
    outFile = [tempname, '.png'];
    c = onCleanup(@() delete_if_exists(outFile));

    % --- 1) Basic call (explicit params) ---
    quadriga_lib.write_png(outFile, gradient, 'jet', 0.0, max_val, false);
    assertTrue(exist(outFile, 'file') == 2);
    delete_if_exists(outFile);

    % --- 2) With one (dummy) output ---
    t = quadriga_lib.write_png(outFile, gradient, 'jet', 0.0, max_val, false);
    assertEqual(t, 1.0);
    assertTrue(exist(outFile, 'file') == 2);
    delete_if_exists(outFile);

    % --- 3) Minimal args (defaults are applied inside MEX) ---
    quadriga_lib.write_png(outFile, gradient);
    assertTrue(exist(outFile, 'file') == 2);
    delete_if_exists(outFile);

    % --- 4) Error: wrong number of inputs ---
    f = @() quadriga_lib.write_png();
    assertExceptionThrown(f, 'quadriga_lib:CPPerror');
end

function delete_if_exists(p)
    if exist(p, 'file') == 2
        try
            delete(p);
        catch
            % ignore
        end
    end
end
