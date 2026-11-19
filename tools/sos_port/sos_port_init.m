% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function res_dir = sos_port_init()
% Adds the quadriga-lib package to the path and returns the folder for the results.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', '..'));
res_dir = fullfile(here, 'results');
if ~exist(res_dir, 'dir'), mkdir(res_dir); end
if ~exist(fullfile(res_dir, 'tables'), 'dir'), mkdir(fullfile(res_dir, 'tables')); end
end
