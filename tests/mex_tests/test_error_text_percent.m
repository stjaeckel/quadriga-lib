function test_error_text_percent
% Error messages that contain text from the caller or from a file (file names, material names) must reach
% MATLAB / Octave unchanged. A '%' in that text is plain text, not a printf directive: '%s' or '%n' must
% neither crash the session nor be replaced in the message.

% A file name with printf directives: the reader fails because the file does not exist
fn = 'missing_%s%n%d.obj';
if exist(fn, 'file')
    delete(fn);
end
check_error(@() quadriga_lib.obj_file_read(fn), 0, fn);

% A material name with printf directives that is not in the material table, read with csv_strict = true.
% The material lookup runs only when the material index output (9th) is requested.
fn = 'error_text_percent.obj';
mtl = '%s%s%s%s%s%s%s%s%s%s%s%s%n';
f = fopen(fn, 'w');
fprintf(f, '%s\n', 'o box');
fprintf(f, '%s\n', 'v 0 0 0');
fprintf(f, '%s\n', 'v 1 0 0');
fprintf(f, '%s\n', 'v 0 1 0');
fprintf(f, '%s\n', ['usemtl ', mtl]);
fprintf(f, '%s\n', 'f 1 2 3');
fclose(f);
check_error(@() quadriga_lib.obj_file_read(fn, '', true), 9, mtl);
delete(fn);

end

function check_error(fun, n_out, expected_text)
raised = false;
try
    if n_out == 0
        fun();
    else
        out = cell(1, n_out);
        [out{:}] = fun();
    end
catch ME
    raised = true;
    assertEqual(ME.identifier, 'quadriga_lib:CPPerror');
    if isempty(strfind(ME.message, expected_text))
        error('moxunit:exceptionNotRaised', '%s', ['EXPECTED text "', expected_text, '" in message, GOT: "', ME.message, '"']);
    end
end
assertTrue(raised);
end
