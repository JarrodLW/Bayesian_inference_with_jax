% matlab script for solving heat equation IBVP with inhomogeneous Dirichlet boundary conditions
% based on https://www.chebfun.org/examples/disk/HeatEqn.html

%function []=cooling_problem(arg)

%test_points = load('/Users/jlw31/PycharmProjects/DETI/Example/'+arg+'.mat');
test_points = load('/Users/jlw31/PycharmProjects/DETI/Example/value_communicating_file.mat');

u0 = 2 + diskfun.harmonic(0,2) + diskfun.harmonic(1,1);
alpha = 0.01;
mu = 1.;
dt = 0.01;                                  % Time step
tfinal = 2;                                 % Stopping time
nsteps = ceil(tfinal/dt);                   % Number of time steps
m = 20;                                     % Spatial discretization

cos_coeffs = test_points.cos_coeffs;
sin_coeffs = test_points.sin_coeffs;
[num_points, num_cos_coefficients] = size(cos_coeffs);
[num_points, num_sin_coefficients] = size(sin_coeffs);
[num_points, domain_dim] = size(test_points.x_vals);
Dirichlet_energies = [];

for ind=1:num_points

    % Constructing the boundary function as a Fourier series
    g = diskfun(@(theta, r) 1);

    for k=1:num_cos_coefficients
        a_coeff = double(cos_coeffs(ind, k))
        g = g + diskfun(@(theta, r) a_coeff*cos(k*theta));
    end

    for l=1:num_sin_coefficients
        b_coeff = double(sin_coeffs(ind, l))
        g = g + diskfun(@(theta, r) b_coeff*sin(l*theta));
    end

    %g = diskfun(@(theta, r) 1 + a_1*cos(theta) + b_1*sin(theta)...
    %+ a_2*cos(2*theta) + b_2*sin(2*theta), 'polar');

    initial_bc = u0(:,1);
    up = u0;                                    % Previous time step

    % One step of backward Euler
    K = sqrt(1/(dt*alpha))*1i;         % Helmholtz frequency for BDF1
    u = diskfun.helmholtz(K^2*up, K, initial_bc, m, m);
    K = sqrt(3/(2*dt*alpha))*1i;       % Helmholtz frequency for BDF2
    for n = 2:nsteps
        t = (n-1)*dt;
        bc = g(:, 1) + exp(-mu.^2*t)*(u0(:, 1) - g(:, 1));
        rhs = K^2/3*(4*u - up);
        up = u;
        u = diskfun.helmholtz(rhs, K, bc, m, m);
        % Plot the solution every 50 time steps
        %{
        if (mod(n, 50) == 0 )
            contourf(u), colormap(hot), caxis([-1, 3])
            title(sprintf('Time %1.2f',n*dt)), colorbar, axis('off'), snapnow
            %plot(u)
            %axis('off'), colormap(jet), colorbar
        end
        %}
    end

    %{
    subplot(1,2,1)
    plot(u0)
    caxis manual
    caxis([0., 4.])
    axis('off'), colormap(jet), colorbar

    subplot(1,2,2)
    plot(u)
    caxis manual
    caxis([0., 4.])
    axis('off'), colormap(jet), colorbar
    %}

    Dirichlet_energy_t0 = norm(grad(u0));
    Dirichlet_energy = norm(grad(u));
    Dirichlet_energies(ind) = Dirichlet_energy;

end
save('/Users/jlw31/PycharmProjects/DETI/Example/value_communicating_file.mat',...
'Dirichlet_energies', 'Dirichlet_energy_t0', '-append')
