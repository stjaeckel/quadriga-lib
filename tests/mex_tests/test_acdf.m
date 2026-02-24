function test_acdf()
% MOxUnit tests for quadriga_lib.acdf

% --- Basic single column ---
data = (0:9).';
[ Sh, bins, Sc, mu, sig ] = quadriga_lib.acdf( data );

assertEqual( numel(bins), 201 );
assertElementsAlmostEqual( bins(1), 0.0, 'absolute', 1e-10 );
assertElementsAlmostEqual( bins(end), 9.0, 'absolute', 1e-10 );

assertEqual( size(Sh), [201, 1] );
assertElementsAlmostEqual( Sh(end, 1), 1.0, 'absolute', 1e-10 );

% CDF should be non-decreasing
assertTrue( all( diff(Sh(:,1)) >= 0 ) );

% Sc should equal Sh for single column
assertEqual( numel(Sc), 201 );
assertElementsAlmostEqual( Sc, Sh(:,1), 'absolute', 1e-10 );

assertEqual( numel(mu), 9 );
assertEqual( numel(sig), 9 );

% sig should be all zeros for single column
assertElementsAlmostEqual( sig, zeros(9,1), 'absolute', 1e-10 );

% --- Multiple identical columns ---
data = [ (0:99).', (0:99).' ];
[ Sh, bins, Sc, mu, sig ] = quadriga_lib.acdf( data );

assertEqual( size(Sh), [201, 2] );
assertElementsAlmostEqual( Sh(:,1), Sh(:,2), 'absolute', 1e-10 );
assertElementsAlmostEqual( Sc(end), 1.0, 'absolute', 0.02 );

% sig reflects variation within quantile window, not across sets
for i = 1:9
    assertTrue( sig(i) < 1.0 );
end

% --- Custom bins ---
data = (0:99).';
custom_bins = [0, 25, 50, 75, 99];
[ Sh, bins ] = quadriga_lib.acdf( data, custom_bins );

assertEqual( numel(bins), 5 );
assertEqual( size(Sh), [5, 1] );
assertElementsAlmostEqual( Sh(end, 1), 1.0, 'absolute', 1e-10 );

% --- Handles Inf and NaN ---
data = [0; 1; 2; 3; 4; 5; 6; 7; 8; 9; Inf; NaN];
[ Sh, bins ] = quadriga_lib.acdf( data );

assertElementsAlmostEqual( bins(1), 0.0, 'absolute', 1e-10 );
assertElementsAlmostEqual( bins(end), 9.0, 'absolute', 1e-10 );
assertElementsAlmostEqual( Sh(end, 1), 1.0, 'absolute', 1e-10 );

% --- Custom n_bins ---
data = (0:99).';
[ Sh, bins ] = quadriga_lib.acdf( data, [], uint64(51) );

assertEqual( numel(bins), 51 );
assertEqual( size(Sh, 1), 51 );

% --- Quantile correctness ---
data = (0:999).';
[ ~, ~, ~, mu ] = quadriga_lib.acdf( data );

assertEqual( numel(mu), 9 );
for q = 1:9
    expected = q * 0.1 * 999.0;
    assertTrue( abs( mu(q) - expected ) < 10.0 );
end

% --- Constant data ---
data = 5 * ones(100, 1);
[ Sh, bins ] = quadriga_lib.acdf( data );

assertEqual( numel(bins), 201 );
assertElementsAlmostEqual( Sh(end, 1), 1.0, 'absolute', 1e-10 );

% --- Error: empty data ---
try
    quadriga_lib.acdf( [] );
    error('Expected an error for empty data');
catch
    % Expected
end

% --- Error: n_bins too small ---
try
    quadriga_lib.acdf( (0:9).', [], uint64(1) );
    error('Expected an error for n_bins=1');
catch
    % Expected
end

end
