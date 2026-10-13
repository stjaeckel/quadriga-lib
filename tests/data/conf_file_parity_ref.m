% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.
%
% Reference for the test case "Diagnostic - conf_file_read parity with QuaDRiGa" in
% tests/catch2_tests/test_conf_file_parity.cpp.
%
% Loads every scenario of QuaDRiGa-NG with the MATLAB parser (qd_builder) and writes the parsed values
% with 17 significant digits to 'conf_file_parity_ref.txt'. QuaDRiGa-NG is only read.
%
% Usage (MATLAB or Octave), from a folder that contains no conf files:
%   run('/sjc/quadriga-lib-conf/tests/data/conf_file_parity_ref.m')
%
% Line formats of the output:
%   list <scenario> <file name>             one line per entry of supported_scenarios(1)
%   scenario <name>                         starts the block of one conf file
%   chk <0 or 1>                            lsp_xcorr_chk
%   sp <key> <value>                        numeric field of scenpar
%   sps <key> <text>                        text field of scenpar
%   plm <text>                              plpar.model
%   pl <key> <n> <value 1> ... <value n>    numeric field of plpar
%   xc <64 values>                          lsp_xcorr, column-major

quadriga_src = '/sjc/QuaDRiGa-NG/quadriga_src';
out_file = fullfile(fileparts(mfilename('fullpath')), 'conf_file_parity_ref.txt');

addpath(quadriga_src);
fid = fopen(out_file, 'w');

[names, files] = qd_builder.supported_scenarios(1);
for n = 1:numel(names)
    fprintf(fid, 'list %s %s\n', names{n}, files{n});
end

scen = qd_builder.supported_scenarios(0);
for n = 1:numel(scen)
    b = qd_builder(scen{n});
    fprintf(fid, 'scenario %s\n', scen{n});
    fprintf(fid, 'chk %d\n', b.lsp_xcorr_chk);

    sp = b.scenpar;
    keys = fieldnames(sp);
    for k = 1:numel(keys)
        v = sp.(keys{k});
        if ischar(v)
            fprintf(fid, 'sps %s %s\n', keys{k}, v);
        else
            fprintf(fid, 'sp %s %.17g\n', keys{k}, v);
        end
    end

    pl = b.plpar;
    if ~isempty(pl)
        keys = fieldnames(pl);
        for k = 1:numel(keys)
            v = pl.(keys{k});
            if ischar(v)
                fprintf(fid, 'plm %s\n', v);
            else
                fprintf(fid, 'pl %s %d%s\n', keys{k}, numel(v), sprintf(' %.17g', v));
            end
        end
    end

    fprintf(fid, 'xc%s\n', sprintf(' %.17g', b.lsp_xcorr));
end
fclose(fid);
fprintf('Wrote %d scenarios to %s\n', numel(scen), out_file);
