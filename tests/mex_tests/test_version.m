% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_version
% MEX interface test for quadriga_lib.version.

v = quadriga_lib.version();
assertTrue( ischar(v) && ~isempty(v) );

% Errors use the common identifier quadriga_lib:CPPerror
failed = false;
try
    [~] = quadriga_lib.version(1);
catch ME
    failed = strcmp(ME.identifier, 'quadriga_lib:CPPerror') && ...
        ~isempty(strfind(ME.message, 'Wrong number of input arguments.'));
end
assertTrue( failed );

failed = false;
try
    [~, ~] = quadriga_lib.version();
catch ME
    failed = strcmp(ME.identifier, 'quadriga_lib:CPPerror') && ...
        ~isempty(strfind(ME.message, 'Wrong number of output arguments.'));
end
assertTrue( failed );

end
