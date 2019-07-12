%% Home assignment 1: Sequential Monte Carlo-based mobility tracking in cellular networks
% Sofia Larsson och Hanna Kerek
clear all
close all
clc

% ------------------------------- First Assignment -----------------------------------%

% Defining constants
sigma = 0.5;
alpha = 0.6;
dt = 0.5;
m = 100; % Arbitrary length of the markov chain

% Defining matrices
psi_tw = [dt^2/2 dt 1]';
psi_tz = [dt^2/2 dt 0]';
phi_t = [1 dt dt^2/2; 0 1 dt; 0 0 alpha];

psi_w = [psi_tw zeros(3,1); zeros(3,1) psi_tw];
psi_z = [psi_tz zeros(3,1); zeros(3,1) psi_tz];
phi = [phi_t zeros(3,3); zeros(3,3) phi_t]; 

P = (1/20)*(ones(5,5)+diag(15*ones(1,5))); 

% Defining distributions
X0 = normrnd(zeros(6,1), [500 5 5 200 5 5]');
Z = [0 0; 3.5 0; 0 3.5; 0 -3.5; -3.5 0]';

X = ones(6,m);
X(:,1) = X0;

index = randi(length(Z));

for i = 1:m-1
    W = normrnd(zeros(2,1),[sigma^2 sigma^2]');
    X(:,i+1) = phi*X(:,i) + psi_z*Z(:,index) + psi_w*W;
    index = randsample([1 2 3 4 5], 1, true, P(index,:));    
end

figure(1)
plot(X(1,:), X(4,:))
title('Expected trajectory of car', 'FontSize', 15)
xlabel('X^1 [m]', 'FontSize', 15)
ylabel('X^2 [m]', 'FontSize', 15)
set(gca,'FontSize',15)
legend('Expected trajectory', 'FontSize', 15)

% ------------------------------- Third Assignment -----------------------------------%

% Loading relevant files 
load stations.mat
load RSSI-measurements.mat

% Defining constants
N = 10000;
m = length(Y); 
eta = 3;
sigma2 = 1.5;
v = 90;

% Creating emtpy matrices for tau and omega
tau = zeros(2,m);
omega = zeros(N,m);

% Defining probability density function
prob = @(x,X) mvnpdf(x,v-10*eta*log10(dis(X,pos_vec,N)),diag(ones(1,6)*sigma2^2));

% Samling the first states of X
index = zeros(N,m);
index(:,1) = randi(length(Z),[N 1]);

% Setting the intital values of X, omega and tau
X = [mvnrnd(zeros(1,6),diag([500;5;5;200;5;5]), N) Z(:,index(:,1))']; % Nx8
omega(:,1) = prob(Y(:,1)',X);
tau(1,1) = sum(X(:,1).*omega(:,1))/sum(omega(:,1));
tau(2,1) = sum(X(:,4).*omega(:,1))/sum(omega(:,1));

% Performing the sequential sampling algorithm
[weights, exp_trajectory, indicies] = seqsampling(prob, omega,tau, X, Y, Z, index);

figure(2)
plot(exp_trajectory(1,:), exp_trajectory(2,:))
hold on 
scatter(pos_vec(1,:), pos_vec(2,:),'*')
title('Expected trajectory with sequential sampling', 'FontSize', 15)
xlabel('X^1 [m]', 'FontSize', 15)
ylabel('X^2 [m]', 'FontSize', 15)
set(gca,'FontSize',15)
legend('Expected trajectory','Stations', 'FontSize', 15)

figure(3)
histogram(log10(weights(:,1)),83,'FaceColor','c')
title('Histogram of the weights at n = 1', 'FontSize', 15)
xlabel('Logarithm of weights', 'FontSize', 15)
set(gca,'FontSize',15)
ylabel('Counts', 'FontSize', 15)

figure(4)
histogram(log10(weights(:,10)),83,'FaceColor','y')
title('Histogram of the weights at n = 10', 'FontSize', 15)
xlabel('Logarithm of weights', 'FontSize', 15)
set(gca,'FontSize',15)
ylabel('Counts', 'FontSize', 15)

figure(5)
histogram(log10(weights(:,40)),83,'FaceColor','g')
title('Histogram of the weights at n = 40', 'FontSize', 15)
xlabel('Logarithm of weights', 'FontSize', 15)
set(gca,'FontSize',15)
ylabel('Counts', 'FontSize', 15)

% Calculating efficient sample size
ESS = zeros(1,4);
j = 1;
for i = [1 10 20 60 80]
    CV = (1/N)*sum(((N * weights(:,i)./sum(weights(:,i))) - 1).^2);
    ESS(j) = (N/(1+CV));
    j = 1+j;
end

fprintf('Efficient sample size for number of iterations')
array2table(ESS, 'VariableNames', {'One','Ten','Twenty','Sixty','Eighty'})

% ------------------------------- Fourth Assignment -----------------------------------%

clear all 

load stations.mat
load RSSI-measurements.mat

% Defining constants
N = 10000;
m = length(Y); 
eta = 3;
sigma2 = 1.5;
v = 90;

% Defining Z states
Z = [0 0; 3.5 0; 0 3.5; 0 -3.5; -3.5 0]';

% Creating emtpy matrices for tau and omega
tau = zeros(2,m);
omega = zeros(N,m);

% Defining probability density function
prob = @(x,X) mvnpdf(x,v-10*eta*log10(dis(X,pos_vec,N)),diag(ones(1,6)*sigma2^2));

% Samling the first states of X
index = zeros(N,m);
index(:,1) = randi(length(Z),[N 1]);

% Setting the intital values of X, omega and tau
X = [mvnrnd(zeros(1,6),diag([500;5;5;200;5;5]), N) Z(:,index(:,1))']; % Nx8
omega(:,1) = prob(Y(:,1)',X);
tau(1,1) = sum(X(:,1).*omega(:,1))/sum(omega(:,1));
tau(2,1) = sum(X(:,4).*omega(:,1))/sum(omega(:,1));

% Performing the sequential sampling with resampling algorithm
[weights, exp_trajectory, indicies] = seqresampling(prob,omega,tau, X, Y, Z, index);

figure(6)
plot(exp_trajectory(1,:), exp_trajectory(2,:))
hold on 
scatter(pos_vec(1,:), pos_vec(2,:),'*')
title('Expected trajectory with sequential resampling', 'FontSize', 15)
xlabel('X^1 [m]', 'FontSize', 15)
ylabel('X^2 [m]', 'FontSize', 15)
set(gca,'FontSize',15)
legend('Expected trajectory','Stations', 'FontSize', 15)

% Calculating the most probable driving command 
sum_weights = zeros(m,5);
prob_drive = zeros(5,m);
for i = 1:5
    sum_weights(:,i) = [indicies == i]'*omega/sum(omega);
end

sum_weights = sum_weights';

[max2, idx] = max(sum_weights);

figure(7)
scatter(1:m,idx)
title('Most probable driving command', 'FontSize', 15)
xlabel('Estimate', 'FontSize', 15)
ylabel('Driving command', 'FontSize', 15)
yticks([1 2 3 4 5])
yticklabels({'None', 'East', 'North', 'South', 'West'})
set(gca,'FontSize',15)

% ------------------------------- Fifth Assignment -----------------------------------%
clear all 

load stations.mat
load RSSI-measurements-unknown-sigma.mat

% Defining constants
N = 10000;
m = length(Y); 
eta = 3;
sigma = 1.6:0.1:3;
v = 90;

% Defining Z states
Z = [0 0; 3.5 0; 0 3.5; 0 -3.5; -3.5 0]';
max_like = zeros(1,length(sigma));
i = 1;

for sig = sigma
    
    % Defining probability density function
    prob = @(x,X) mvnpdf(x,v-10*eta*log10(dis(X,pos_vec,N)),diag(ones(1,6)*sig^2));
    
    % Creating emtpy matrices for tau and omega
    tau = zeros(2,m);
    omega = zeros(N,m);
    
    % Samling the first Z state
    index = zeros(N,m);
    index(:,1) = randi(length(Z),[N 1]);

    % Setting the intital values of X, omega and tau
    X = [mvnrnd(zeros(1,6),diag([500;5;5;200;5;5]), N) Z(:,index(:,1))']; % Nx8
    omega(:,1) = prob(Y(:,1)',X);
    tau(1,1) = sum(X(:,1).*omega(:,1))/sum(omega(:,1));
    tau(2,1) = sum(X(:,4).*omega(:,1))/sum(omega(:,1));

    % Performing the sequential sampling with resampling algorithm
    [weights, ~,~] = seqresampling(prob,omega,tau, X, Y, Z, index);    
    
    % Calculate the maximum likelihood of weights
    max_like(i) = sum(log(sum(weights)));
    i = i + 1;
end

figure(8)
scatter(sigma,max_like)
title('Maximum likelihood of variance \zeta', 'FontSize', 15)
xlabel('\zeta', 'FontSize', 15)
ylabel('f(y_{0:m})', 'FontSize', 15)
set(gca,'FontSize',15)

[max_eta, idx] = max(max_like);

fprintf('The best zeta is %d', sigma(idx))

%------------------ plot the traectory with the best sigma ----------------

sig = sigma(idx);

% Defining probability density function
prob = @(x,X) mvnpdf(x,v-10*eta*log10(dis(X,pos_vec,N)),diag(ones(1,6)*sig^2));

% Creating emtpy matrices for tau and omega
tau = zeros(2,m);
omega = zeros(N,m);

% Samling the first Z state
index = zeros(N,m);
index(:,1) = randi(length(Z),[N 1]);

% Setting the intital values of X, omega and tau
X = [mvnrnd(zeros(1,6),diag([500;5;5;200;5;5]), N) Z(:,index(:,1))']; % Nx8
omega(:,1) = prob(Y(:,1)',X);
tau(1,1) = sum(X(:,1).*omega(:,1))/sum(omega(:,1));
tau(2,1) = sum(X(:,4).*omega(:,1))/sum(omega(:,1));

% Performing the sequential sampling with resampling algorithm
[weights, exp_trajectory, indicies] = seqresampling(prob,omega,tau, X, Y, Z, index);

figure(9)
plot(exp_trajectory(1,:), exp_trajectory(2,:))
hold on 
scatter(pos_vec(1,:), pos_vec(2,:),'*')
title('Expected trajectory with \zeta = 2.2', 'FontSize', 15)
xlabel('X^1 [m]', 'FontSize', 15)
ylabel('X^2 [m]', 'FontSize', 15)
legend('Expected trajectory','Stations', 'FontSize', 15)
set(gca,'FontSize',15)