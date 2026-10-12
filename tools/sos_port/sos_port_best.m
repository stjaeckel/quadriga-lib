function [best, all] = sos_port_best(type, n_dim, recipe, N, seeds)
% Returns the table with the best error on the coordinate planes (mse_core) among the seeds, and all of them.
if nargin < 5, seeds = 1:4; end
for k = 1:numel(seeds)
    t = sos_port_table(type, n_dim, recipe, N, seeds(k));
    if k == 1, all = t; else, all(k) = t; end
end
[~, i] = max([all.mse_core]);
best = all(i);
end
