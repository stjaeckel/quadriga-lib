function test_get_channels_irs()
% Comprehensive MOxUnit tests for quadriga_lib.get_channels_irs
% Single test function; no test_suite created.

% ---------- Common fixtures ----------

% Antenna configs
n_el=3; n_az=4; n_elem_tx=2; n_ports_tx=2;
n_elem_rx=2; n_ports_rx=2;
n_elem_irs=2; n_ports_irs=2;

ant_tx = make_ant(n_el,n_az,n_elem_tx,n_ports_tx,1.0);                  % isotropic-like, real
ant_rx = make_ant(n_el,n_az,n_elem_rx,n_ports_rx,1.0);
ant_irs = make_ant(n_el,n_az,n_elem_irs,n_ports_irs,1.0);               % default IRS
ant_irs_diff = make_ant(n_el,n_az,n_elem_irs,n_ports_irs,0.3);          % for asymmetry / i_irs effect

% Geometry (meters) & orientations (radians)
tx_pos=[0;0;1]; rx_pos=[20;10;1]; irs_pos=[10;0;5];
tx_ori=[0;0;0]; rx_ori=[0;0;0]; irs_ori=[0;0;0];

% Scatterers (two paths each leg)
fbs_pos_1 = [ 2  4; 0 1; 3 4 ];
lbs_pos_1 = [ 6  8; 0 1; 4 5 ];
fbs_pos_2 = [12 15; 1 2; 5 6 ];
lbs_pos_2 = [16 18; 5 6; 4 3 ];

% Path lengths consistent with geometry
path_length_1 = zeros(1,2);
for k=1:2
    path_length_1(k)=norm(tx_pos-fbs_pos_1(:,k))+norm(fbs_pos_1(:,k)-lbs_pos_1(:,k))+norm(lbs_pos_1(:,k)-irs_pos);
end
path_length_2 = zeros(1,2);
for k=1:2
    path_length_2(k)=norm(irs_pos-fbs_pos_2(:,k))+norm(fbs_pos_2(:,k)-lbs_pos_2(:,k))+norm(lbs_pos_2(:,k)-rx_pos);
end

% Gains (linear)
path_gain_1 = [1.0, 0.2];
path_gain_2 = [1.0, 0.05];

% Polarization transfer matrices (real -> no phase when f0=0)
M_1 = zeros(8,2);
M_1([1,3,5,7],:) = 1;
M_2 = M_1;

% Convenience
all_mask = true(1, numel(path_gain_1)*numel(path_gain_2));
some_mask = [true true true false]; % length 4

% ---------- 1) Basic call; all paths kept; f0=0 ----------
i_irs = 0; thr_keep_all = -300; f0_zero = 0; use_abs=false; active_path_in=[];
[cr0,ci0,D0,act0,aod0,eod0,aoa0,eoa0] = quadriga_lib.get_channels_irs( ...
    ant_tx, ant_rx, ant_irs, ...
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, M_1, ...
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, M_2, ...
    tx_pos, tx_ori, rx_pos, rx_ori, irs_pos, irs_ori, ...
    i_irs, thr_keep_all, f0_zero, use_abs, active_path_in, [] );

% Sizes
assertEqual(size(cr0), [n_ports_rx, n_ports_tx, 4]);
assertEqual(size(ci0), [n_ports_rx, n_ports_tx, 4]);
assertEqual(size(D0),  [n_ports_rx, n_ports_tx, 4]);
assertEqual(size(aod0),[n_ports_rx, n_ports_tx, 4]);
assertEqual(size(eod0),[n_ports_rx, n_ports_tx, 4]);
assertEqual(size(aoa0),[n_ports_rx, n_ports_tx, 4]);
assertEqual(size(eoa0),[n_ports_rx, n_ports_tx, 4]);
assertEqual(numel(act0), 4);
assertTrue(islogical(act0));

% Coefficients: imaginary part ~ 0 when center_freq==0 and real patterns/M
assertTrue(max(abs(ci0(:))) < 1e-12);
assertTrue(any(abs(cr0(:))>0));

% Angles finite, within expected ranges
assertTrue(all(isfinite(aod0(:)) & isfinite(eod0(:)) & isfinite(aoa0(:)) & isfinite(eoa0(:))));
assertTrue(all(aod0(:)>=-pi & aod0(:)<=pi));
assertTrue(all(aoa0(:)>=-pi & aoa0(:)<=pi));
assertTrue(all(eod0(:)>=-pi/2 & eod0(:)<=pi/2));
assertTrue(all(eoa0(:)>=-pi/2 & eoa0(:)<=pi/2));

% ---------- 2) Phase enabled when center_freq>0 ----------
f0 = 3.5e9;
[~,ci1,~,~,~,~,~,~] = quadriga_lib.get_channels_irs( ...
    ant_tx, ant_rx, ant_irs, ...
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, M_1, ...
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, M_2, ...
    tx_pos, tx_ori, rx_pos, rx_ori, irs_pos, irs_ori, ...
    i_irs, thr_keep_all, f0, use_abs, active_path_in, [] );
assertTrue(any(abs(ci1(:))>1e-8));

% ---------- 3) Absolute delays add a constant offset ----------
use_abs=true;
[~,~,D_abs,~,~,~,~,~]= quadriga_lib.get_channels_irs( ...
    ant_tx, ant_rx, ant_irs, ...
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, M_1, ...
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, M_2, ...
    tx_pos, tx_ori, rx_pos, rx_ori, irs_pos, irs_ori, ...
    i_irs, thr_keep_all, f0, use_abs, active_path_in, [] );
use_abs=false;
[~,~,D_rel,~,~,~,~,~]= quadriga_lib.get_channels_irs( ...
    ant_tx, ant_rx, ant_irs, ...
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, M_1, ...
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, M_2, ...
    tx_pos, tx_ori, rx_pos, rx_ori, irs_pos, irs_ori, ...
    i_irs, thr_keep_all, f0, use_abs, active_path_in, [] );
delta = D_abs - D_rel;
assertTrue(all(abs(delta(:)-delta(1))<1e-12));  % same offset everywhere

% ---------- 4) active_path selection overrides threshold ----------
thr_drop_all = +100; % would drop everything if used
[cr_sel,ci_sel,~,act_sel] = quadriga_lib.get_channels_irs( ...
    ant_tx, ant_rx, ant_irs, ...
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, M_1, ...
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, M_2, ...
    tx_pos, tx_ori, rx_pos, rx_ori, irs_pos, irs_ori, ...
    i_irs, thr_drop_all, f0_zero, false, some_mask, [] );
assertEqual(size(cr_sel,3), sum(some_mask));
assertEqual(size(ci_sel,3), sum(some_mask));
assertEqual(numel(act_sel), numel(some_mask));
assertTrue(islogical(act_sel));
assertEqual(sum(act_sel), sum(some_mask));
assertTrue(all(~act_sel | some_mask(:))); % used paths must be a subset of requested

% ---------- 5) i_irs index changes coefficients when IRS has multiple ports ----------
i_irs0 = 0;
i_irs1 = 1;
[cr_i0,~,~,~,~,~,~,~] = quadriga_lib.get_channels_irs( ...
    ant_tx, ant_rx, ant_irs, ...
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, M_1, ...
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, M_2, ...
    tx_pos, tx_ori, rx_pos, rx_ori, irs_pos, irs_ori, ...
    i_irs0, thr_keep_all, f0_zero, false, all_mask, [] );
[cr_i1,~,~,~,~,~,~,~] = quadriga_lib.get_channels_irs( ...
    ant_tx, ant_rx, ant_irs, ...
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, M_1, ...
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, M_2, ...
    tx_pos, tx_ori, rx_pos, rx_ori, irs_pos, irs_ori, ...
    i_irs1, thr_keep_all, f0_zero, false, all_mask, [] );
assertTrue(any(abs(cr_i0(:)-cr_i1(:))>1e-12));

% ---------- 6) Asymmetric IRS (ant_irs_2) changes result ----------
[cr_sym,~,~,~,~,~,~,~] = quadriga_lib.get_channels_irs( ...
    ant_tx, ant_rx, ant_irs, ...
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, M_1, ...
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, M_2, ...
    tx_pos, tx_ori, rx_pos, rx_ori, irs_pos, irs_ori, ...
    i_irs0, thr_keep_all, f0_zero, false, all_mask, [] );
[cr_asym,~,~,~,~,~,~,~] = quadriga_lib.get_channels_irs( ...
    ant_tx, ant_rx, ant_irs, ...
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, M_1, ...
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, M_2, ...
    tx_pos, tx_ori, rx_pos, rx_ori, irs_pos, irs_ori, ...
    i_irs0, thr_keep_all, f0_zero, false, all_mask, ant_irs_diff );
assertTrue(any(abs(cr_sym(:)-cr_asym(:))>1e-12));

% ---------- 8) Error handling: mismatched M_1 size ----------
M_1_bad = ones(8,1); % should be 8x2
assertExceptionThrown(@() quadriga_lib.get_channels_irs( ...
    ant_tx, ant_rx, ant_irs, ...
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, M_1_bad, ...
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, M_2, ...
    tx_pos, tx_ori, rx_pos, rx_ori, irs_pos, irs_ori, ...
    i_irs0, thr_keep_all, f0_zero, false, all_mask, [] ));

% ---------- helpers ----------
    function ant = make_ant(nel,naz,nelem,nports,scale)
        ant.e_theta_re = ones(nel,naz,nelem)*scale;
        ant.e_theta_im = zeros(nel,naz,nelem);
        ant.e_phi_re   = ones(nel,naz,nelem)*scale;
        ant.e_phi_im   = zeros(nel,naz,nelem);
        ant.azimuth_grid   = linspace(-pi,pi,naz);
        ant.elevation_grid = linspace(-pi/2,pi/2,nel);
        ant.element_pos = [linspace(0,(nelem-1)*0.5,nelem); zeros(1,nelem); zeros(1,nelem)];
        % simple coupling: element k -> port k (or scaled second port for IRS variety)
        C = zeros(nelem,nports);
        for k=1:min(nelem,nports), C(k,k)=1; end
        % Add slight port scaling so i_irs effect is observable
        if nports>=2, C(:,2)=C(:,2)*0.5*scale; end
        ant.coupling_re = C;
        ant.coupling_im = zeros(size(C));
    end
end
