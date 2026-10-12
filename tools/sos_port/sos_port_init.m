function res_dir = sos_port_init()
% Adds the quadriga-lib package to the path and returns the folder for the results.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', '..'));
res_dir = fullfile(here, 'results');
if ~exist(res_dir, 'dir'), mkdir(res_dir); end
if ~exist(fullfile(res_dir, 'tables'), 'dir'), mkdir(fullfile(res_dir, 'tables')); end
end
