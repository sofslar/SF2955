function tnew = t_draw(t,d,i, rho)
    R = zeros(1,d-1);
    
    for j=2:d
       R(j-1) = rho*(t(j+1,i)-t(j-1,i)); 
    end
    
    thelp = sort(t(2:d,i)' + (-R + 2*R.*rand(1,d-1)));
    
    if (thelp <= t(end)) + (thelp >= t(1)) == 2*ones(1,d-1)
      tnew = [t(1) thelp t(end)]; 
    else
      tnew = t_draw(t,d,i,rho);
    end  
end