% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_write_png
% MEX interface test for quadriga_lib.write_png.
% Physics validation lives in the Catch2 suite (test_write_png.cpp).

    %% Setup: gradient test data, as in the Python test
    N = 50;
    max_val = 10.0;
    v = linspace(0.0, max_val, N+1); % mimic endpoint=False
    v(end) = [];
    gradient = repmat(v, 25, 1);
    gradient(1, :) = 0.0;
    gradient(end, :) = 10.0;

    outFile = [tempname, '.png'];
    c = onCleanup(@() delete_if_exists(outFile));

    %% Basic call, explicit parameters
    quadriga_lib.write_png(outFile, gradient, 'jet', 0.0, max_val, false);
    assertTrue(exist(outFile, 'file') == 2);
    delete_if_exists(outFile);

    %% Error: the function returns no output
    try
        t = quadriga_lib.write_png(outFile, gradient, 'jet', 0.0, max_val, false);
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    catch ME
        expectedErrorMessage = 'Wrong number of output arguments.';
        if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
            error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
        end
    end
    assertFalse(exist(outFile, 'file') == 2);

    %% Minimal arguments: defaults
    quadriga_lib.write_png(outFile, gradient);
    assertTrue(exist(outFile, 'file') == 2);
    delete_if_exists(outFile);

    %% Error: infinite limits (NaN selects the data minimum / maximum)
    f = @() quadriga_lib.write_png(outFile, gradient, 'jet', NaN, Inf);
    assertExceptionThrown(f, 'quadriga_lib:CPPerror');
    assertFalse(exist(outFile, 'file') == 2);

    %% Error: wrong number of inputs
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
