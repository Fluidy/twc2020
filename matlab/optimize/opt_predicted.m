function opt_predicted(env_name, exp_name)
%% Read predictor parameters
predictor_dir = ['../../experiments/', exp_name, '/model'];
predictor_params = jsondecode(fileread([predictor_dir, '/predictor_config.json']));
obs_window = predictor_params.obs_window;
pre_window = predictor_params.pre_window;

%% Read environment parameters
data_dir = ['../../data/', env_name];
env_params = jsondecode(fileread([data_dir, '/env_config.json']));
delta_T = env_params.delta_T;
seg_TF = env_params.seg_TF;  % number of time frames per segment
num_seg = env_params.num_seg;  % number segments
bandwidth = env_params.bandwidth;

total_TF = seg_TF*(num_seg - 1); % total number of TF to be optimized

%% Loading predicted SNR
load([predictor_dir, '/predicted_snr.mat'], 'predicted_snr', 'true_snr', 'pre_predict');  
load([data_dir, '/test_data.mat'], 'seg_size_data')  
load('../func_fitting/p.mat', 'p');

predicted_snr = double(predicted_snr);  % convert tensorflow single precision to double for fmincon
num_ep = size(predicted_snr, 1);

%% Optimize average rate allocation for each TF
R = zeros(num_ep, total_TF);
for i = 1 : num_ep
    TF_optimized = 0;
    seg_transmitted = 1; % number of segments transmitted
    
    
    disp(['optimizing episode: ', num2str(i)])

    if pre_predict == 0
        TF_opt = obs_window;
        seg_size_opt = seg_size_data(i, seg_transmitted + 1 : seg_transmitted + TF_opt/seg_TF)';
        R(i, TF_optimized + 1:TF_optimized + TF_opt) = kron(seg_size_opt/seg_TF/delta_T, ones(seg_TF, 1));
        
        TF_optimized = TF_optimized + TF_opt;  % 1:pre_window using non-predictive transmission
        seg_transmitted = seg_transmitted + TF_opt/seg_TF;
    end
    
    predicted_snr_used = 0;
    while TF_optimized < total_TF
        TF_opt = min(total_TF - TF_optimized, pre_window); % number of TF to optimize
        SNR_dB = squeeze(predicted_snr(i, predicted_snr_used + 1 : predicted_snr_used + TF_opt));     
        SNR = 10.^(SNR_dB/10);
        
        if mod(TF_opt, seg_TF) == 0
            % number of segment to be transmitted in the current optimization step
            seg_trans = TF_opt/seg_TF;  
        else
            error('The optimize window should be divisible by seg_TF');   
        end
        
        seg_size_opt = seg_size_data(i, seg_transmitted + 1 : seg_transmitted + seg_trans)';
        x0 = kron(seg_size_opt/seg_TF/delta_T, ones(seg_TF, 1))*log(2)/bandwidth;
        A = kron(tril(ones(seg_trans)), ones(1, seg_TF));
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
        
        R(i, TF_optimized + 1:TF_optimized + TF_opt) = x*bandwidth/log(2);
        
        
        TF_optimized = TF_optimized + TF_opt;
        seg_transmitted = seg_transmitted + seg_trans;
        predicted_snr_used = predicted_snr_used + TF_opt;
    end

end

loss = std(predicted_snr(:) - true_snr(:));
save([predictor_dir, '/planned_actions.mat'], 'R');

%%
% i = 11;
% close all;
% figure
% plot(cumsum(R(i, :))*delta_T); hold on
% stairs(seg_TF:seg_TF:(num_seg - 1)*seg_TF, cumsum(seg_size_data(i, 2:end)));
end
