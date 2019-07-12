function distance = dis(X, pos_vec,N)
    n = length(pos_vec);
    dist = zeros(n,N);
    for i = 1:n
        dist(i,:) = vecnorm([X(:,1) X(:,4)]'-pos_vec(:,i));
    end
    distance = dist';
end