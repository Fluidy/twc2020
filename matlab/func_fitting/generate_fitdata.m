clearvars;
W = 20; % MHz 
R = linspace(1e-8, 100, 1000); % Mbps
x_min = 1e-8;
x_max = 100;

x = R/W*log(2);
y = zeros(size(x));

if max(R) > -ei(-x_min)*W/log(2)
    error('x_min too large')
end
if min(R) < -ei(-x_max)*W/log(2)
    error('x_max too small')
end

parfor i = 1:length(R)
    y(i) = exp(-inverse_e1(x(i), [x_min, x_max]))/inverse_e1(x(i), [x_min, x_max]);
    disp(i)
end

save('fitdata', 'x', 'y', 'W', 'R');
