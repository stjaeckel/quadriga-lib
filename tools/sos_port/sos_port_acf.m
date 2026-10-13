function acf = sos_port_acf(type, dist, d_lambda)
% Target ACF of the paper: 'Exp' (2), 'Gauss', 'Comb' (22), for the decorrelation distance d_lambda.
x = dist / d_lambda;
switch type
    case 'Exp',   acf = exp(-x);
    case 'Gauss', acf = exp(-x.^2);
    case 'Comb',  acf = exp(-x.^2); acf(x > 1) = exp(-x(x > 1));
    otherwise, error('Unknown ACF type.');
end
end
