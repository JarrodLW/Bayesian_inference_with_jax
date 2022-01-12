% based on https://www.chebfun.org/examples/disk/HeatEqn.html

test_points = load('/Users/jlw31/PycharmProjects/DETI/Example/Test.mat');

tic 
%%{
coeffs = test_points.x_vals;
a_1 = double(coeffs(1));
b_1 = double(coeffs(2));
a_2 = double(coeffs(3));
b_2 = double(coeffs(4));
%%}

Dirichlet_energy = a_1 + b_1 + a_2 + b_2
toc

save('/Users/jlw31/PycharmProjects/DETI/Example/Test.mat', 'Dirichlet_energy')
