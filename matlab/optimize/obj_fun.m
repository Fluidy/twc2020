function [y, g] = obj_fun(x, p, SNR)
% Compute the objection function y (i.e., total transmit energy consumption)
% and its gradient g

x = x(:); 
SNR = SNR(:);
p = p(:);
z = x.^(0:length(p) - 1)*p;
y = sum(1./SNR.*(z - x));
g = 1./SNR.*((x.^(0 : length(p)-2) * (p(2 : end).*(1 : length(p)-1)')) - 1);
end

