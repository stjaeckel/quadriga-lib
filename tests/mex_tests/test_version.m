function test_version

v = quadriga_lib.version();
assertTrue( ischar(v) && ~isempty(v) );

% Errors use the common identifier quadriga_lib:CPPerror
failed = false;
try
    [~] = quadriga_lib.version(1);
catch ME
    failed = strcmp(ME.identifier, 'quadriga_lib:CPPerror') && ...
        ~isempty(strfind(ME.message, 'Incorrect number of input arguments.'));
end
assertTrue( failed );

failed = false;
try
    [~, ~] = quadriga_lib.version();
catch ME
    failed = strcmp(ME.identifier, 'quadriga_lib:CPPerror') && ...
        ~isempty(strfind(ME.message, 'Incorrect number of output arguments.'));
end
assertTrue( failed );

end
