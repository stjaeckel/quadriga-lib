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

try
    quadriga_lib.subdivide_triangles;
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    tro = quadriga_lib.subdivide_triangles(rand(3,3));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    tro = quadriga_lib.subdivide_triangles(rand(3,3,'single'));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

end
