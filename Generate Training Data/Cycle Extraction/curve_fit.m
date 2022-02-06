% Curve Fitting - Use non-linear equation solver lsqnonlin
% See [1] GD Clifford et al. Model-based filtering, compression and classification of the ECG
% [2] GD Clifford A NOVEL FRAMEWORK FOR SIGNAL REPRESENTATION AND SOURCE SEPARATION: APPLICATIONS TO FILTERING AND SEGMENTATION OF BIOSIGNALS

%% Load Data

clearvars

load 'extracted_cycles_cleaned.mat'

fitting_parameter = struct('id', 0 , 'a', [],'b', [],'c', [], 'rmse', []);
%% Curve Fitting
parfor i = 1:length(extracted_cycles)
    cycles = extracted_cycles(i).cycles;
    
    a_i = [];
    b_i = [];
    c_i = [];
    rmse_i = [];
    
    for j = 1:length(cycles)
        recg = cycles(j).data(end,:);
        fs = cycles(j).fs;
        t = linspace(0, 2*pi, length(recg));
        RR = length(recg) / fs;
        alpha = sqrt(1/RR);
        
        rpos = round(cycles(j).rr(1)/3);
        r_val = recg(rpos);
        % P Q R S T+ T-
        init_a = [1.2; -5.0;30.0;-7.5;alpha^(1.5)/2;3*alpha^(1.5)/2;];
        init_theta = ([-0.14*sqrt(alpha); -0.01*alpha;0;0.03*alpha;sqrt(alpha)*0.23;sqrt(alpha)*0.25]) *2*pi + (2/3)*pi;
        init_b = [0.2*alpha;0.1*alpha;0.1*alpha;0.1*alpha;0.4*(1/alpha);0.2*alpha];
        init_a = init_a .* init_b .* init_b;
        
        upper_a = [max(recg);0;max(recg);0;max(recg);max(recg)];
        upper_theta = [1.2;(2/3)*pi;2*pi;3;2*pi-1;2*pi-1];
        upper_b = 2*pi*ones(6,1);
        
        lower_a = [0;min(recg);min(recg);min(recg);0;0];
        lower_theta = [0;1.5;0;(2/3)*pi;(2/3)*pi;(2/3)*pi];
        lower_b = zeros(6,1);
        
        ub = [upper_a;upper_theta;upper_b];
        lb = [lower_a;lower_theta;lower_b];
        
        % Constraints of theta Ax <= b
        A = [0 0 0 0 0 0 1 -1 0 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 1 -1 0 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 1 -1 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 1 -1 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 1 -1 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 -1 0;
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 -1;
             0 0 0 0 0 0 0 0 0 0 -1 1 0 0 0 0 0 0;
             0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0;];
        b = [-0.1;-0.1;-0.1;-0.1;-0.1;0;0;0.5;-1.5];
        
        init_params = [init_a;init_theta;init_b];

        refx = recg;
        fun = @(p) norm(err_func(t ,refx, p), 2)^2;
        options = optimoptions('fmincon','Display','off');

        if j == 1
            [est_params, sqr_err] = fmincon(fun, init_params, A, b, [], [], lb, ub, [], options);
        else
            [est_params, sqr_err] = fmincon(fun, init_params2, A, b, [], [], lb, ub, [], options);
        end

        est_params = reshape(est_params,[],3);
        a_i(j,:) = est_params(:,1)'; % a
        b_i(j,:) = est_params(:,2)'; % theta
        c_i(j,:) = est_params(:,3)'; % b
        init_params2 = est_params;
        rmse_i(j) = sqrt(sqr_err / length(recg));
    end
    fitting_parameter(i).id = i;
    fitting_parameter(i).a = a_i;
    fitting_parameter(i).b = b_i;
    fitting_parameter(i).c = c_i;
    fitting_parameter(i).rmse = rmse_i;
end

save('cycle_parameters_gauss6_4_0.mat', 'fitting_parameter')


% z = \sum{ai * exp(-(1/2) * (dthetai/bi)^2)}, dthetai = t - thetai
function z = gauss_model(t, a, theta, b)
dtheta = rem(t - theta, 2*pi);
sum_term = a .* exp(-(dtheta.^2)./(2*b.^2));
z = sum(sum_term);
end

% params = [a1;... ;ai ;theta1;...thetai; b1...bi;]
function err = err_func(t , refx, params)
params = reshape(params,[],3);
a = params(:,1);
theta = params(:, 2);
b = params(:, 3);
isolevel = 0;
z = gauss_model(t, a, theta, b) + isolevel;

err = refx - z;
end