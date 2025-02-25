function test_channel_export_obj_file

fn = 'test.obj';
if exist( fn,'file' )
    delete(fn);
end
if exist( 'test.mtl','file' )
    delete('test.mtl');
end

rx_pos = rand(3,1);
tx_pos = rand(3,1);
coeff_re = rand(3,2,5,3);
coeff_im = rand(3,2,5,3);
no_interact = ones(5,3);
interact_coord = rand(3,5,3);
center_freq = 3.0e9;

quadriga_lib.channel_export_obj_file( fn, [], [], [], [] ,[], [], [], ...
    rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );

quadriga_lib.channel_export_obj_file( fn, 0, [], [], [] ,[], [], [], ...
    rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );

quadriga_lib.channel_export_obj_file( fn, 1, [], [], [] ,[], [], [], ...
    rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );

quadriga_lib.channel_export_obj_file( fn, [], 10, 6, [] ,[], [], [], ...
    rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );

% Test ill-formatted file name
try
    quadriga_lib.channel_export_obj_file( '', [], [], [], [] ,[], [], [], ...
        rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'OBJ-File name must end with .obj';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Test wrong colormap
try
    quadriga_lib.channel_export_obj_file( fn, [], [], [], 'bls' ,[], [], [], ...
        rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Colormap is not supported.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Test wrong number of edges
try
    quadriga_lib.channel_export_obj_file( fn, [], [], [], 'parula' ,[], [], 1, ...
        rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of edges mut be >= 3.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% This should create objects that only contain paths without a volume
quadriga_lib.channel_export_obj_file( fn, [], [], [], [] , 0, [], [], ...
    rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );

% Test negative radius
try
    quadriga_lib.channel_export_obj_file( fn, [], [], [], [] , -1, [], [], ...
        rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Radius cannot be negative.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Test ill-formatted rx position
try
    quadriga_lib.channel_export_obj_file( fn, [], [], [], [] , [], [], [], ...
        rand(2,1), tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = '''rx_pos'' is missing or ill-formatted (must have 3 rows).';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Test ill-formatted interact_coord
try
    quadriga_lib.channel_export_obj_file( fn, [], [], [], [], [], [], [], ...
        rand(2,1), tx_pos, no_interact, interact_coord(:,:,[1,2]), center_freq, coeff_re, coeff_im  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of snapshots in interact_coord must match coefficients.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

quadriga_lib.channel_export_obj_file( fn, [], 10, 6, [] ,[], [], [], ...
    rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im, [1,1]  );

% Test ill-formatted interact_coord
try
    quadriga_lib.channel_export_obj_file( fn, [], 10, 6, [] ,[], [], [], ...
        rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im, 0  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Snapshot indices ''i_snap'' cannot exceed the number of snapshots in the channel.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

delete(fn);
delete('test.mtl');
