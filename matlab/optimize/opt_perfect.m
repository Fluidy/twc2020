function opt_perfect(env_name, exp_name)
%% Read environment parameters
data_dir = ['../../data/', env_name];
env_params = jsondecode(fileread([data_dir, '/env_config.json']));
delta_T = env_params.delta_T;
seg_TF = env_params.seg_TF;  % number of time frames per segment
num_seg = env_params.num_seg;  % number segments
bandwidth = env_params.bandwidth;
playback_start = env_params.playback_start;
total_TF = seg_TF*(num_seg - 1); % total number of TF to be optimized

%% Loading predicted SNR
load([data_dir, '/test_data.mat'], 'seg_size_data', 'snr_data') % get seg_size_data
load('../func_fitting/p.mat', 'p') ;  % get objection function fitting results

num_ep = size(snr_data, 1);

%% Optimize average rate allocation for each TF
R = zeros(num_ep, total_TF);
energy = zeros(num_ep, 1);
for i = 1 : num_ep
   
    disp(['optimizing episode: ', num2str(i)])

    SNR_dB = snr_data(i, playback_start + 1 : playback_start + total_TF);     
    SNR = 10.^(SNR_dB/10);


    seg_size_opt = seg_size_data(i, 2 : end)';
    x0 = kron(seg_size_opt/seg_TF/delta_T, ones(seg_TF, 1))*log(2)/bandwidth;
    A = kron(tril(ones(num_seg - 1)), ones(1, seg_TF));
    b = cumsum(log(2)/bandwidth/delta_T * seg_size_opt);
    lb = zeros(length(x0), 1);
    ub = [];


    options = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
        'SpecifyObjectiveGradient', true, ...
        'HessianFcn', @(x, lambda)hess_fun(x, p, SNR, lambda), ...,
        'OptimalityTolerance', 1e-6, ...,
        'StepTolerance', 1e-6, ...,
        'Display', 'off');
    [x, fval] = fmincon(@(x)obj_fun(x, p, SNR), x0, -A, -b, [], [], lb, ub, [], options);

    R(i, :) = x*bandwidth/log(2);
    energy(i) = fval;
end

results_dir =  ['../../experiments/', exp_name, '/model'];
mkdir(results_dir)
save([results_dir, '/planned_actions.mat'], 'R');
end
