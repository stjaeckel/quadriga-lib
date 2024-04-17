function test_subdivide_triangles

tri = rand(12,9);

tro = quadriga_lib.subdivide_triangles(tri);
assertEqual( tri, tro );

tro = quadriga_lib.subdivide_triangles(tri,2);
assertEqual( size(tro), [48,9] );
assertEqual( tri(1,1:3), tro(1,1:3) );
assertEqual( tri(1,1:3), tro(1,1:3) );

tros = quadriga_lib.subdivide_triangles(single(tri),2);
assertTrue( isa(tros,'single') );
assertElementsAlmostEqual( single(tro), tros, 'absolute', 1e-7 );

mtli = rand(12,5);
[tros, mtlo] = quadriga_lib.subdivide_triangles(tri,2, mtli);

% 0 outputs
quadriga_lib.subdivide_triangles(tri,2, mtli);


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
    expectedErrorMessage = 'Too many output arguments.';
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

try % wrong mtl prop
    [~,~] = quadriga_lib.subdivide_triangles(tri,2, mtli(1,:));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''triangles_in'' and ''mtl_prop'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end
