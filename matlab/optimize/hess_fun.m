function H = hess_fun(x, p, SNR, ~)
% Compute the Hessian matrix of the objective function
x = x(:);
SNR = SNR(:);
p = p(:);
H_diag = 1./SNR.*(x.^(0:length(p) - 3) * (p(3:end).*(2:length(p)-1)'));
H = diag(H_diag);
end

