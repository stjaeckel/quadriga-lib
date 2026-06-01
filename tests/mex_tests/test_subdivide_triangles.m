function test_subdivide_triangles

tri = rand(12,9);

tro = quadriga_lib.subdivide_triangles(tri);
assertEqual( tri, tro );

tro = quadriga_lib.subdivide_triangles(tri,2);
assertEqual( size(tro), [48,9] );
assertEqual( tri(1,1:3), tro(1,1:3) );
assertEqual( tri(1,1:3), tro(1,1:3) );

tros = quadriga_lib.subdivide_triangles(single(tri),2);
assertTrue( isa(tros,'double') );
assertElementsAlmostEqual( single(tro), tros, 'absolute', 1e-7 );

mtli = uint64(randi(5, 12, 1));   % 1-based per-face material index
[tros, mtlo] = quadriga_lib.subdivide_triangles(tri, 2, mtli);

assertEqual( size(mtlo), [48, 1] );
assertEqual( mtlo(1:4), repmat(mtli(1), 4, 1) );   % parent 1 -> 4 sub-triangles
assertEqual( mtlo(5:8), repmat(mtli(2), 4, 1) );   % parent 2 -> 4 sub-triangles

% 0 outputs
quadriga_lib.subdivide_triangles(tri, 2, mtli);


try % 5 imputs
    quadriga_lib.subdivide_triangles(tri,2, mtli, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 0 imputs
    quadriga_lib.subdivide_triangles;
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 3outputs
    [~,~,~] =  quadriga_lib.subdivide_triangles(tri,2, mtli);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    tro = quadriga_lib.subdivide_triangles(rand(3,3));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''triangles_in'' must have 9 columns.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong mtl_ind length
    [~,~] = quadriga_lib.subdivide_triangles(tri, 2, mtli(1));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of triangles in ''triangles_in'' and length of ''mtl_ind'' do not match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end
