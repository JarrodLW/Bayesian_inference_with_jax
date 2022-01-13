% matlab script for solving heat equation IBVP with inhomogeneous Dirichlet boundary conditions, for arbitrary order
% of Fourier truncation
% based on https://www.chebfun.org/examples/disk/HeatEqn.html

test_points = load('/Users/jlw31/PycharmProjects/DETI/Heat_Transfer_Example/value_communicating_file_2.mat');

%u0 = 2 + diskfun.harmonic(0,2) + diskfun.harmonic(1,1);
u0 = diskfun(@(x,y) 2 + exp(-5*(x+.4).^2 -5*(y+.2).^2) + 0.8* exp(-10*(x-.5).^2 -10*(y+.6).^2) ...
+1.2*exp(-20*(x-.5).^2 -20*(y-.3).^2));
alpha = 0.1;
mu = 10.;
dt = 0.01;                                  % Time step
tfinal = 2;                                 % Stopping time
nsteps = ceil(tfinal/dt);                   % Number of time steps
m = 20;                                     % Spatial discretization


phase_list = test_points.phases;
[num_points, domain_dim] = size(test_points.x_vals);
obs = [];

for ind=1:num_points

    phase = double(phase_list(ind));
    g =  diskfun(@(theta, r) 1 + 0.5*cos(theta+phase));
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

    %{
    Dirichlet_energy_t0 = norm(grad(u0));
    Dirichlet_energy = norm(grad(u));
    Dirichlet_energy_diff = Dirichlet_energy_t0 - Dirichlet_energy;
    Dirichlet_energy_diffs(ind) = Dirichlet_energy_diff;
    %}

    max_t0 = max2(u0);
    max_t1 = max2(u);
    obs(ind) = max_t0 - max_t1;

end
save('/Users/jlw31/PycharmProjects/DETI/Heat_Transfer_Example/value_communicating_file_2.mat',...
'obs', 'max_t0', '-append')
