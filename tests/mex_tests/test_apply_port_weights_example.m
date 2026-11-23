% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_apply_port_weights_example
% MEX interface test for quadriga_lib.apply_port_weights: combines a 4-element dipole array
% (smoke test, no assertions).

ant = quadriga_lib.generate_antenna('dipole');

% Duplicate 4 times
ant.e_theta_re  = repmat(ant.e_theta_re, [1,1,4]);
ant.e_theta_im  = repmat(ant.e_theta_im, [1,1,4]);
ant.e_phi_re    = repmat(ant.e_phi_re, [1,1,4]);
ant.e_phi_im    = repmat(ant.e_phi_im, [1,1,4]);
ant.element_pos = repmat(ant.element_pos, [1,4]);

% Set element positions and port weights
ant.element_pos(2,:) = [ -0.75, -0.25, 0.25, 0.75];  % lambda, along y-axis
ant.port_weights_re = [ 1 ; 1 ; 1 ; 1 ]/sqrt(4);
ant.port_weights_im = [ 0 ; 0 ; 0 ; 0 ];

ant_c = quadriga_lib.apply_port_weights( ant );


end
