function x = inverse_e1(y, x_range)
% Compute the inverse function of E1(x) defined in our paper
x = fzero(@(x)(ei(-x) + y), x_range); 
end

