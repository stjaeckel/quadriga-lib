function test_icospere

quadriga_lib.icosphere;

center = quadriga_lib.icosphere;
assertEqual( size(center), [20,3] );

center2 = quadriga_lib.icosphere([],2);
assertElementsAlmostEqual( 2*center, center2, 'absolute', 1e-14 );

[center, length] = quadriga_lib.icosphere(1,3);
assertElementsAlmostEqual( sqrt(sum(center.^2,2)), length, 'absolute', 1e-14 );

[center, ~, vert] = quadriga_lib.icosphere(2);
assertElementsAlmostEqual( sum((center + vert(:,1:3)).^2,2), ones(80,1), 'absolute', 1e-14 );
assertElementsAlmostEqual( sum((center + vert(:,4:6)).^2,2), ones(80,1), 'absolute', 1e-14 );
assertElementsAlmostEqual( sum((center + vert(:,7:9)).^2,2), ones(80,1), 'absolute', 1e-14 );

[~, ~, ~, dir] = quadriga_lib.icosphere(2);
assertTrue( all(all(abs(dir(:,[1,3,5]) )<=pi)) );
assertTrue( all(all(abs(dir(:,[2,4,6]) )<=pi/2)) );

try
    [~,~,~,~,~] = quadriga_lib.icosphere;
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    quadriga_lib.icosphere(1,2,3);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

end
