clear all

p = ones(1,181);
for n = 1:180
    p(n+1) = p(n)*exp(1j*pi*n/180);
end
p = p .* (10+(1:181)/10);
p = [p,(p(end-1:-1:1))];

z = zeros(size(p));

az_grd = -180:1:180;
el_grd = 0;

az = -179:0.1:179;
el = zeros(size(az));

[vr,vi,hr,hi] = arrayant_lib.interpolate( real(p),imag(p),z,z, az_grd*pi/180 ,el_grd, az*pi/180, el );
pii = vr + 1j*vi;

figure(1)
plot( az_grd, abs(p),'--r','Linewidth',2 )
hold on
plot( az, abs(pii),'-k' )
hold off

figure(2)
plot( az_grd(1:end-1), diff(unwrap(angle(p))*180/pi),'--r','Linewidth',2 )
hold on
plot( az(1:end-1), 10*diff(unwrap(angle(pii))*180/pi),'-k' )
hold off



