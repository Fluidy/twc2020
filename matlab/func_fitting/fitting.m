close all;
clearvars;
load('fitdata.mat'); % load x, y, W, R


d_min = 100;
d_max = sqrt(200^2 + 250^2);
distance = d_min : 1 : d_max;

alpha_dB = - 35.3 - 37.6*log10(distance);
sigma2_dB = -174 + 10*log10(W*1e6) + 9 - 30; % in dB
SNR_dB = alpha_dB - sigma2_dB;
SNR = 10.^(SNR_dB/10);


p = 1/SNR(1)*(y - x);
figure; hold on;
plot(R, p, '+')



[Xdata, Ydata] = prepareCurveData(x, y);


fitresult = fit(Xdata, Ydata, 'poly8', 'Weight', 1./Ydata);
p = [fitresult.p9;
     fitresult.p8;
     fitresult.p7;
     fitresult.p6;
     fitresult.p5;
     fitresult.p4;
     fitresult.p3;
     fitresult.p2;
     fitresult.p1;];

R = linspace(0, 200, 1000);  % Mbps
x = R/W*log(2);
x = x(:);  % make sure x is a column vector
y_fit = max(0, x.^(0:length(p) - 1)*p);

p_fit = 1/SNR(1)*(y_fit - x);

plot(R, p_fit)
xlim([0, 200])

save('p', 'p');