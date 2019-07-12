%% Statistical inference from coal mine disaster and mixture model data 
% using Markov chain Monte Carlo and the EM-algorithm
% 2/5-2019

%% Baysian analysis of coal mine disasters - constructing a complex MCMC algorithm

clear all
close all
clc

% Loading relevant data
load coal-mine.csv

% Defining constants
M = 4000;
burn_in = 10000; 
d = 4;  % d-1 breakpoints
rho = 0.05;
v = 20;

% Defining t 
tstart = 1851;
tend = 1963;
t = zeros(d+1,burn_in + M);
t(1,:) = tstart; 
t(d+1,:) = tend;
t(2:d,1) = sort(tstart+rand(1,d-1)*(tend-tstart));

% Allocation
acc_rate = zeros(1,M+burn_in);

% summing over the disasters
[N, edges] = histcounts(coal_mine, t(:,1));

% hyperparameters
lambda = zeros(d-1,burn_in+M);

% parameters
theta = gamrnd(2,1/v); 
lambda = gamrnd(2*ones(d,1), ones(d,1)/theta);

% t conditional distribution
f = @(t, l, N) exp(-diff(t)*l)*prod(diff(t))*prod(l'.^N);

for i = 1:burn_in + M-1 

    % Draw from distribuacction
    lambda(:,i) = gamrnd(2 + N, 1./(diff(t(:,i))'+ theta));
    theta = gamrnd(2+2*d,1/(v+sum(lambda(:,i))));

    % Draw new beakpoints with Random walk proposal and the condition t1<t2<t3.. 
    tnew = t_draw(t,d,i,rho);

    % Calculate the number of disasters in each subinterval
    [Nnew, edges] = histcounts(coal_mine, tnew);

    % Calculate the condition
    func = f(tnew,lambda(:,i), Nnew)/f(t(:,i)',lambda(:,i), N);

    % Metropolis- Hasting step
    if rand < min(1,func) 
        t(:,i+1) = tnew;
        N = Nnew;
        acc_rate(i) = 1;
    else 
        t(:,i+1) = t(:,i); 
    end  
end

figure(1)
plot(burn_in:burn_in + M, t(:,burn_in:end))
title('Four breakpoints', 'FontSize', 15)
xlabel('Iteration', 'FontSize', 15)
ylabel('Year', 'FontSize', 15)
set(gca,'FontSize',15)
legend('Start year','1st BP','2nd BP','3rd BP', 'End year', 'FontSize', 15)

%----------- EM-based inference in mixture models ---------------%
clear all

% Loading relevant data 
load mixture-observations.csv

% defining constants
N = 100;
theta = 0.5;
y = mixture_observations;
n = length(y);
w = zeros(n,N);

%-------------------- Plotting the histogram --------------------%
figure(2)
histogram(y, 'FaceColor', 'y')
title('Mixture Observation', 'FontSize', 15)
xlabel('Observation', 'FontSize', 15)
ylabel('Counts', 'FontSize', 15)
set(gca,'FontSize',15)

track_theta = zeros(1,N);

%------------------------ EM algorithm --------------------------%

for i = 1:N
    y0 = normpdf(y,0,1);
    y1 = normpdf(y,1,2);  
    w(:,i) = theta.*y1./(theta.*y1 + (1-theta).*y0); 
    theta = sum(w(:,i))/n;
    track_theta(i) = theta;
end

%------------------------ Plotting theta -----------------------%

figure(3)
plot(1:N,track_theta,'LineWidth',2)
title('Convergence of EM algorithm', 'FontSize', 15)
xlabel('Iteration', 'FontSize', 15)
ylabel('\theta', 'FontSize', 15)
set(gca,'FontSize',15)



