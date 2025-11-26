clear all
clc

ant = quadriga_lib.arrayant_generate('custom', 1, 2.4e9, 6, 10);
ant = quadriga_lib.arrayant_copy_element(ant, 1, 2:36);
for n = 2 : 36
    ant = quadriga_lib.arrayant_rotate_pattern(ant,0,0,(n-1)*10, 0,n);
end

of = ones(4,2) * 100;
c = quadriga_lib.get_channels_ieee_indoor(ant, ant, 'A', 2.4, 10, 2, [], [], [], [], 1.99, [], [], of, 100, 11 );
P1 = c(1).coeff_re(:,:,2).^2 + c(1).coeff_im(:,:,2).^2; 
P2 = c(2).coeff_re(:,:,2).^2 + c(2).coeff_im(:,:,2).^2; 

figure(1)
bar(sum(P1,1));

figure(2)
bar(sum(P1,2));

figure(3)
bar(sum(P2,1));

figure(4)
bar(sum(P2,2));


 
% W = pow(:,:);
% A = aod(:,:);
% 
% AS_rms = sqrt( sum(W.*(A.^2),1) ./ sum(W,1) ); % 1×M
% 
% stem(aod(:,1), pow(:,1))
% 
% P = 10*log10(squeeze(sum(pow,1)));
% imagesc(P)


n_users = 6;
seed_AoD_LOS = 2803;
range_AoD_LOS = 360;
rand('seed',seed_AoD_LOS);
offsets_AoD_LOS = (rand(n_users,1)-0.5)*range_AoD_LOS