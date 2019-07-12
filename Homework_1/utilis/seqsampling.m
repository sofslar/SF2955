
function [weights, exp_trajectory, indicies] = seqsampling(prob, omega,tau, X, Y, Z, index)
    sample = ones(length(X),1).*[1 2 3 4 5];
    [N,m] = size(omega);
    dt = 0.5;
    alpha = 0.6;
    sigma = 0.5;
    
    % Defining matrices for the chain
    psi_tw = [dt^2/2 dt 1]';
    psi_tz = [dt^2/2 dt 0]';
    phi_t = [1 dt dt^2/2; 0 1 dt; 0 0 alpha];

    psi_w = [psi_tw zeros(3,1); zeros(3,1) psi_tw];
    psi_z = [psi_tz zeros(3,1); zeros(3,1) psi_tz];
    phi = [phi_t zeros(3,3); zeros(3,3) phi_t];
    
    for n = 2:m     
        % Updating the Markox indices 
        z = X(:,7:8);
        [~,oldindex] = ismember(z, Z', 'rows'); 
        matris = [ones(N,15).*oldindex sample];
        index(:,n) = matris(sub2ind(size(matris),(1:N)',randi(20,N,1)));
       
        % Sequential sampling
        W = mvnrnd(zeros(2,1),[sigma^2 sigma^2],N);
        X = [[(phi*X(:,1:6)' + psi_z*X(:,7:8)' + psi_w*W')]' Z(:,index(:,n))'];
        omega(:,n) = prob(Y(:,n)',X).*omega(:,n-1);
 
        % Calculating the expected trajectory
        tau(1,n) = sum(X(:,1).*omega(:,n))/sum(omega(:,n));
        tau(2,n) = sum(X(:,4).*omega(:,n))/sum(omega(:,n));
    end  
    
    exp_trajectory = tau; 
    weights = omega; 
    indicies = index;
end