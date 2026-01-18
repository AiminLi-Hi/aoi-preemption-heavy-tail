%% run_benchmark_simulation.m
% Monte Carlo Verification & Baseline Comparison for Lomax AoI Control
% (Precision V5.0: Full Policy Map, Strict Consistency, Cross-Validation)
% -------------------------------------------------------------------------
% KEY FEATURES:
% 1. HJB Solver: Outputs full mappings z*(Delta) and tau*(y).
% 2. Simulation: Implements strict grid quantization and table lookup.
% 3. Validation: Separates Optimization CRN from Evaluation CRN.
% -------------------------------------------------------------------------
% Author: AImin Li
% Date: 2025-01-11

clear; 

%% 1. Setup Parameters & HJB Solver
% -------------------------------------------------------------------------
fprintf('=== Step 1: Solving HJB for Optimal Policy (Full Map) ===\n');

% --- System Parameters ---
kappa_s = 1;        % Sampling cost
kappa_p = 5;        % Preemption penalty
alpha   = 2.1;

% Lomax tail index
kappa   = 1;        % Lomax scale
dt      = 0.005;    % Grid step (Keep it fine)
yCut    = 30;       % State truncation

% --- Solver Parameters ---
eps_tail = 1e-6;    
maxIter  = 60;      
tolV     = 1e-8;    
tolRho   = 1e-10;   
verbose  = false;   
forbidTauZero = true;

% --- Pre-computation ---
slope = kappa/(alpha-1);
tauCap = kappa*(eps_tail^(-1/alpha)-1);
tauCap = floor(tauCap/dt)*dt;

y = (0:dt:yCut)'; M = numel(y);
t = (0:dt:tauCap)'; T = numel(t);

% Kernels
barF = (1 + t/kappa).^(-alpha);
f    = (alpha/kappa) * (1 + t/kappa).^(-alpha-1);
u = 1 + t/kappa;
A  = (kappa/(alpha-1)) * (1 - u.^(1-alpha));
J1 = kappa^2 * ( (u.^(2-alpha)-1)/(2-alpha) - (u.^(1-alpha)-1)/(1-alpha) );

% Candidates
tauFine = 8; nLog = 250;
tauFine = min(tauFine, tauCap);
dense = 0:dt:tauFine;
if tauFine < tauCap
    a = max(tauFine, dt);
    tail = exp(linspace(log(a), log(tauCap), nLog));
    cand = unique([dense, tail, tauCap]);
else
    cand = dense;
end
cand = unique(floor(cand/dt)*dt);
tauCandIdx = unique(round(cand/dt)+1);
if forbidTauZero, tauCandIdx(tauCandIdx==1) = []; end

% --- Init ---
rho = 0; v = zeros(M,1); tauIdx = tauCandIdx(1) * ones(M,1);
rho_hist = nan(maxIter,1);

% --- HJB Loop ---
z_star_map = zeros(M,1); % To store z*(y) for simulation

for it = 1:maxIter
    v_old = v; rho_old = rho; tau_old = tauIdx;
    vCutVal = v(end);
    
    % Step 1: Idle envelope (Construction of z* map)
    phi_grid = kappa_s + v + 0.5*y.^2 - rho*y;
    m_grid = zeros(M,1); 
    arg_grid = zeros(M,1); % Stores index of z*
    
    m_grid(M)=phi_grid(M); arg_grid(M)=M;
    for i=M-1:-1:1
        if phi_grid(i) <= m_grid(i+1)
            m_grid(i)=phi_grid(i); arg_grid(i)=i;
        else
            m_grid(i)=m_grid(i+1); arg_grid(i)=arg_grid(i+1);
        end
    end
    
    % Store the discrete policy map z*(y)
    z_star_map = y(arg_grid); 
    
    % Compute hI for the integral term
    const_ext = kappa_s + vCutVal - slope*yCut;
    hI = zeros(T,1); zChoice = zeros(T,1);
    
    for k=1:T
        Delta = t(k);
        if Delta <= yCut
            i0 = round(Delta/dt)+1;
            m1 = m_grid(i0);
            z1 = y(arg_grid(i0));
        else
            m1 = inf; z1 = inf;
        end
        L = max(Delta, yCut);
        z2 = max(L, rho - slope);
        m2 = 0.5*z2^2 + (slope-rho)*z2 + const_ext;
        if m1 <= m2, zStar = z1; mStar = m1; else, zStar = z2; mStar = m2; end
        zChoice(k) = zStar;
        hI(k) = rho*Delta - 0.5*Delta^2 + mStar;
    end
    
    w = zChoice - t;
    I_fh = cumtrapz(t, f .* hI);
    
    % Step 3: Busy improvement
    for i=1:M
        feas = tauCandIdx;
        if isempty(feas), continue; end
        yi = y(i);
        jj = feas(:);
        tauv = t(jj);
        yplus = yi + tauv;
        v_shift = zeros(size(yplus));
        inside = (yplus <= yCut);
        % Consistent linear interpolation for solver
        if any(inside), v_shift(inside) = interp1(y, v, yplus(inside), 'linear'); end
        if any(~inside), v_shift(~inside) = vCutVal + slope*(yplus(~inside)-yCut); end
        Q = yi .* A(jj) + J1(jj) - rho .* A(jj) + I_fh(jj) + barF(jj).*(kappa_p + v_shift);
        [~,kmin] = min(Q);
        tauIdx(i) = jj(kmin);
    end
    
    % Step 4: Policy Eval
    Bw = cumtrapz(t, f .* w);
    Ccost = cumtrapz(t, f .* (t.*w + 0.5*w.^2 + kappa_s));
    
    usedTau = unique(tauIdx); maxJ = max(usedTau);
    compCoeff = cell(maxJ,1); compConst = cell(maxJ,1);
    cVec = sparse(M,1); cCst = 0;
    
    if ismember(1, usedTau), compCoeff{1}=cVec; compConst{1}=cCst; end
    for j=2:maxJ
        for kk=[j-1, j]
            wt = (dt/2)*f(kk); zt = zChoice(kk);
            if zt <= yCut
                iL = floor(zt/dt)+1; iL = max(1, min(M-1, iL));
                zL = y(iL); zR = y(iL+1);
                lam = (zt - zL)/(zR - zL);
                cVec(iL) = cVec(iL) + wt*(1-lam); cVec(iL+1) = cVec(iL+1) + wt*lam;
            else
                cVec(M) = cVec(M) + wt; cCst = cCst + wt*slope*(zt - yCut);
            end
        end
        if ismember(j, usedTau), compCoeff{j}=cVec; compConst{j}=cCst; end
    end
    
    nn = M + 1; Aeq = spalloc(nn, nn, 40*M); beq = zeros(nn,1);
    for i=1:M
        j = tauIdx(i);
        g_i = A(j) + Bw(j);
        r_i = y(i)*A(j) + J1(j) + Ccost(j) + barF(j)*kappa_p;
        Aeq(i,1) = g_i; Aeq(i,1+i) = 1;
        cc = compCoeff{j}; [rows,~,vals] = find(cc);
        if ~isempty(rows), Aeq = Aeq + sparse(i*ones(numel(rows),1), 1+rows, -vals, nn, nn); end
        beq(i) = r_i + compConst{j};
        
        tauv = t(j); yplus = y(i) + tauv;
        if yplus <= yCut
            iL = floor(yplus/dt)+1; iL = max(1, min(M-1, iL));
            yL = y(iL); yR = y(iL+1);
            lam = (yplus - yL)/(yR - yL);
            Aeq(i,1+iL) = Aeq(i,1+iL) - barF(j)*(1-lam); Aeq(i,1+iL+1) = Aeq(i,1+iL+1) - barF(j)*lam;
        else
            Aeq(i,1+M) = Aeq(i,1+M) - barF(j); beq(i) = beq(i) + barF(j)*slope*(yplus - yCut);
        end
    end
    
% --- Normalization + far-field slope closure (MUST overwrite whole rows) ---
% Replace row M by v(0)=0 (remove Bellman equation at i=M)
    Aeq(M,:) = 0;
    Aeq(M, 1+1) = 1;      % v(y=0)=0  -> v(1)=0
    beq(M) = 0;
    
    % Replace row M+1 by far-field slope closure
    Aeq(M+1,:) = 0;
    Aeq(M+1, 1+M)   = 1;  % v(M)
    Aeq(M+1, 1+M-1) = -1; % v(M-1)
    beq(M+1) = slope*dt;
    
    sol = Aeq\beq;
    rho = sol(1);
    v   = sol(2:end);
    rho_hist(it) = rho;
    
    dv = norm(v - v_old, inf); dr = abs(rho - rho_old); dp = nnz(tauIdx ~= tau_old);
    if dv<=tolV && dr<=tolRho && dp==0, break; end
end
fprintf('HJB Converged. Rho_Theory = %.6f\n', rho);

% --- Pack HJB Policy for Simulation (Strict Table Lookup) ---
hjb_policy.type = 'HJB';
hjb_policy.z_map = z_star_map;       % Vector: z*(y) on grid
hjb_policy.tau_map = t(tauIdx);      % Vector: tau*(y) on grid
hjb_policy.dt = dt;
hjb_policy.yCut = yCut;
hjb_policy.slope = slope;            % For far-field z*
hjb_policy.rho = rho;                % For far-field z*
hjb_policy.M = M;

%% 2. Common Random Numbers (CRN) - Split Sets
% -------------------------------------------------------------------------
fprintf('\n=== Step 2: Generating CRN Sets (Optimization vs Validation) ===\n');
N_OPT = 1e5; 
N_VAL = 5e5; 

% Optimization Set (for Static Search)
U_opt = rand(N_OPT, 1);
S_opt = kappa * (U_opt.^(-1/alpha) - 1);
CRN_Opt.S = S_opt; CRN_Opt.Limit = N_OPT;

% Validation Set (for Head-to-Head)
U_val = rand(N_VAL, 1);
S_val = kappa * (U_val.^(-1/alpha) - 1);
CRN_Val.S = S_val; CRN_Val.Limit = N_VAL;

%% 3. Optimizing Static Policy (on Opt Set)
% -------------------------------------------------------------------------
fprintf('=== Step 3a: Optimizing Static Policy (on Training CRN) ===\n');
d_grid = 0.1:0.1:2.8; 
tau_grid = [0.1:0.1:3.0, 4:1:10]; 
min_cost_static = inf;
best_D = 0; best_T = inf;

for D_val = d_grid
    for T_val = tau_grid
        pol.type = 'Static'; pol.D = D_val; pol.T = T_val;
        [cost_est,~] = simulate_aoi_smdp_strict_q99(kappa_s, kappa_p, pol, CRN_Opt);
        if cost_est < min_cost_static
            min_cost_static = cost_est;
            best_D = D_val; best_T = T_val;
        end
    end
end
fprintf('Best Static Found: Delta=%.2f, Tau=%.2f, Cost(Train)=%.5f\n', ...
        best_D, best_T, min_cost_static);

%% 3b. Solving AoI-Optimal (Update-or-Wait) Policy (on Opt Set)
% -------------------------------------------------------------------------
fprintf('=== Step 3b: Solving AoI-Optimal Update-or-Wait Policy (Paper) ===\n');

% We emulate the paper's "no preemption" setting by taking T=Inf (use 1e9 in code).
% The policy is: wait w = max(beta - Y, 0), i.e., z(y)=max(y,beta).
% Here we solve beta on the OPT CRN set via bisection.

% ---- options ----
T_no_preempt = 1e9;      % effectively disable preemption
fmax = Inf;              % set finite value if you want frequency constraint (paper extension)
Mcap = Inf;              % waiting cap; keep Inf to match your simulator's Static structure
beta_tol = 1e-6;

% training samples of service time Y
S_train = CRN_Opt.S(:);

% solve beta with bisection (toolbox-free, CRN-based expectation)
beta = solve_beta_update_or_wait_crn(S_train, fmax, Mcap, beta_tol);

% pack as a "Static" policy so your simulator can evaluate it directly
aoiopt_policy.type = 'Static';
aoiopt_policy.D    = beta;
aoiopt_policy.T    = T_no_preempt;

% evaluate on training set (optional, for logging)
[cost_aoiopt_train, q99_aoiopt_train] = simulate_aoi_smdp_strict_q99(kappa_s, kappa_p, aoiopt_policy, CRN_Opt);

fprintf('AoI-Optimal (paper) Found: beta=%.6f, T=Inf, Cost(Train)=%.5f, q99(Train)=%.5f\n', ...
        beta, cost_aoiopt_train, q99_aoiopt_train);

%% 4. Final Head-to-Head (on Validation Set)
% -------------------------------------------------------------------------
fprintf('\n=== Step 4: Final Validation (on Independent Test CRN) ===\n');

% HJB Evaluation
[cost_hjb_val, q99_hjb] = simulate_aoi_smdp_strict_q99(kappa_s, kappa_p, hjb_policy, CRN_Val);

% Static Evaluation
static_policy.type = 'Static'; static_policy.D = best_D; static_policy.T = best_T;
[cost_stat_val, q99_stat] = simulate_aoi_smdp_strict_q99(kappa_s, kappa_p, static_policy, CRN_Val);

% Zero Wait Evaluation
zw_policy.type = 'Static'; zw_policy.D = 0; zw_policy.T = T_no_preempt;
[cost_zw_val, q99_zw] = simulate_aoi_smdp_strict_q99(kappa_s, kappa_p, zw_policy, CRN_Val);

% AoI-optimal Evaluation
[cost_aoiopt_val, q99_aoiopt_val] = simulate_aoi_smdp_strict_q99(kappa_s, kappa_p, aoiopt_policy, CRN_Val);


fprintf('---------------------------------------------------------------\n');
fprintf('Policy              | Mean Cost (Test) | q99(AoI, time-weighted)\n');
fprintf('---------------------------------------------------------------\n');
fprintf('HJB Optimal         | %.5f          | %.5f\n', cost_hjb_val,    q99_hjb);
fprintf('AoI-Optimal (paper) | %.5f          | %.5f\n', cost_aoiopt_val, q99_aoiopt_val);
fprintf('Optimized Static    | %.5f          | %.5f\n', cost_stat_val,   q99_stat);
fprintf('Zero-Wait           | %.5f          | %.5f\n', cost_zw_val,     q99_zw);
fprintf('---------------------------------------------------------------\n');


if cost_hjb_val <= cost_stat_val
    fprintf('SUCCESS: HJB consistency restored! It outperforms Static on Test Set.\n');
else
    fprintf('WARNING: Static still better. Check far-field closure or grid resolution.\n');
end

%% 5. Visualization
% -------------------------------------------------------------------------
[~, t_h, a_h] = simulate_aoi_plot(kappa, alpha, kappa_s, kappa_p, hjb_policy, 200);
[~, t_s, a_s] = simulate_aoi_plot(kappa, alpha, kappa_s, kappa_p, static_policy, 200);

figure('Position', [100, 100, 1000, 400], 'Color', 'w');
hold on;
plot(t_s, a_s, 'b--', 'LineWidth', 1.5);
plot(t_h, a_h, 'r-', 'LineWidth', 1.5);
xlabel('Time'); ylabel('Age');
legend('Optimized Static', 'HJB Optimal (Table Lookup)');
title('Sample Path: Strict Consistency Check');
grid on;
%% === Plot v(y) and optimal tau*(y) ===
tau_map = hjb_policy.tau_map;   % tau*(y) on grid [0:yCut]

% 1) v(y)
figure('Color','w','Position',[120 120 900 360]);
plot(y, v, 'LineWidth', 1.6);
grid on;
xlabel('y (busy-start age)');
ylabel('v(y)  (normalized: v(0)=0)');
title(sprintf('Bias function v(y), \\rho = %.6f', rho));

% 2) tau*(y)
figure('Color','w','Position',[120 520 900 360]);
plot(y, tau_map, 'LineWidth', 1.6);
grid on;
xlabel('y (busy-start age)');
ylabel('\\tau^*(y)');
title('Optimal preemption threshold map \\tau^*(y)');


%% ========================================================================
%% Helper 1: Strict Simulation (Lookup Table & Quantization)
%% ========================================================================
function [avg_cost, q99] = simulate_aoi_smdp_strict_q99(kappa_s, kappa_p, policy, CRN)
    % SMDP-consistent simulation + time-weighted AoI 99% quantile
    % - completion -> idle decision, pay kappa_s
    % - preemption -> immediate next busy-start, pay kappa_p
    % - records AoI segments [a,b] for time-weighted quantile

    S_vec = CRN.S; max_ep = CRN.Limit;
    isHJB = strcmp(policy.type, 'HJB');

    if isHJB
        z_map = policy.z_map;
        tau_map = policy.tau_map;
        dt = policy.dt;
        yCut = policy.yCut;
        M = policy.M;
        rho = policy.rho;
        slope = policy.slope;
    else
        D_static = policy.D;
        T_static = policy.T;
    end

    % time & cost
    t_curr = 0;
    total_cost = 0;

    % state
    inIdle = true;
    Delta_idle = 0;
    y_busy = NaN;

    % segment storage (idle and busy each episode => at most 2 segments/ep)
    segA = zeros(2*max_ep,1);
    segB = zeros(2*max_ep,1);
    segN = 0;

    for ep = 1:max_ep
        % ----- IDLE (only after completion) -----
        if inIdle
            if isHJB
                idx = round(Delta_idle/dt) + 1;
                if idx <= M
                    z_target = z_map(idx);
                else
                    z_target = max(Delta_idle, rho - slope);
                end
            else
                z_target = max(Delta_idle, D_static);
            end

            w = max(0, z_target - Delta_idle);

            if w > 0
                % record AoI segment during idle: AoI from Delta_idle to Delta_idle+w
                segN = segN + 1;
                segA(segN) = Delta_idle;
                segB(segN) = Delta_idle + w;
            end

            % cost during idle waiting
            total_cost = total_cost + Delta_idle*w + 0.5*w^2;

            % sampling cost
            total_cost = total_cost + kappa_s;

            t_curr = t_curr + w;
            y_busy = Delta_idle + w;
            inIdle = false;
        end

        % ----- BUSY -----
        if isHJB
            idx = round(y_busy/dt) + 1;
            if idx <= M
                tau = tau_map(idx);
            else
                tau = tau_map(end);  % if you often hit this, increase yCut
            end
        else
            tau = T_static;
        end

        S = S_vec(ep);
        dur = min(S, tau);

        % record AoI segment during busy: AoI from y_busy to y_busy+dur
        if dur > 0
            segN = segN + 1;
            segA(segN) = y_busy;
            segB(segN) = y_busy + dur;
        end

        % cost during busy
        total_cost = total_cost + y_busy*dur + 0.5*dur^2;

        t_curr = t_curr + dur;

        if S <= tau
            % completion: AoI resets to service time S, enter idle
            Delta_idle = S;
            inIdle = true;
        else
            % preemption: pay kappa_p, immediate restart busy with age y+tau
            total_cost = total_cost + kappa_p;
            y_busy = y_busy + dur;
            inIdle = false;
        end
    end

    avg_cost = total_cost / t_curr;

    % ----- time-weighted 99% quantile of AoI -----
    a = segA(1:segN);
    b = segB(1:segN);
    q99 = time_weighted_quantile_from_segments(a, b, 0.99);
end


function q = time_weighted_quantile_from_segments(a, b, p)
    % Each segment contributes uniform time mass over AoI interval [a,b]
    % Time-weighted CDF is proportional to overlap length; quantile can be found exactly by sweep.

    L = b - a;
    valid = L > 0 & isfinite(a) & isfinite(b);
    a = a(valid); b = b(valid); L = L(valid);

    T = sum(L);
    if T <= 0
        q = NaN;
        return;
    end

    n = numel(a);
    x = [a; b];
    typ = [ones(n,1); -ones(n,1)]; % +1 start, -1 end

    % Sort by x; if tie, process end (-1) before start (+1)
    E = [x, typ];
    E = sortrows(E, [1 2]);

    mass = 0;  % accumulated probability mass
    c = 0;     % number of active segments (density multiplier)
    i = 1; m = size(E,1);

    while i <= m
        x0 = E(i,1);

        % apply all events at x0 (ends first due to sorting)
        while i <= m && E(i,1) == x0
            c = c + E(i,2);
            i = i + 1;
        end

        if i > m
            break;
        end

        x1 = E(i,1);
        dx = x1 - x0;

        if dx > 0 && c > 0
            add = (c * dx) / T;
            if mass + add >= p
                q = x0 + (p - mass) * T / c;
                return;
            end
            mass = mass + add;
        end
    end

    % If p is extremely close to 1, return max endpoint
    q = max(b);
end



%% Helper 2: Plotting Sim
function [avg_cost, t_hist, a_hist] = simulate_aoi_plot(kappa, alpha, ks, kp, policy, Tmax)
    % Simplified wrapper for plotting, re-using strict logic structure manually
    t_curr = 0; age = 0; t_hist = [0]; a_hist = [0];
    isHJB = strcmp(policy.type, 'HJB');
    
    while t_curr < Tmax
        % Idle
        if isHJB
            idx = round(age/policy.dt)+1;
            if idx<=policy.M, z=policy.z_map(idx); else, z=max(age, policy.rho-policy.slope); end
            wt = max(0, z-age);
        else
            wt = max(0, policy.D - age);
        end
        
        if wt>0, t_hist(end+1)=t_curr+wt; a_hist(end+1)=age+wt; end
        t_start = t_curr + wt; age_start = age + wt;
        
        % Busy
        if isHJB
            idx = round(age_start/policy.dt)+1;
            if idx<=policy.M, tau=policy.tau_map(idx); else, tau=policy.tau_map(end); end
        else
            tau = policy.T;
        end
        
        U = rand; S = kappa * (U^(-1/alpha) - 1);
        if S <= tau
            dur = S; age = S; p = 0;
        else
            dur = tau; age = age_start + dur; p = kp;
        end
        t_curr = t_start + dur;
        t_hist(end+1) = t_curr; a_hist(end+1) = age;
    end
    avg_cost = 0;
end
%% --- local solver (place it at end of file with your other helpers) ---
function beta = solve_beta_update_or_wait_crn(S, fmax, M, tol)
% Solve beta for Update-or-Wait via bisection using CRN samples S of Y.
% Policy: z(Y) = min(max(beta - Y, 0), M)   (if M=Inf -> (beta-Y)^+)
% Let X = Y + z(Y) (inter-update time in no-preemption regime)
% Paper equation (Algorithm 2): find beta such that
%   E[X] = max( 1/fmax , E[X^2]/(2*beta) )
% If fmax=Inf, becomes E[X] = E[X^2]/(2*beta).

    if nargin < 2 || isempty(fmax), fmax = Inf; end
    if nargin < 3 || isempty(M),    M    = Inf; end
    if nargin < 4 || isempty(tol),  tol  = 1e-6; end

    S = S(:);
    S = S(isfinite(S) & S >= 0);
    if isempty(S), error('solve_beta_update_or_wait_crn: empty S.'); end

    % function o(beta) whose root we seek (monotone in beta)
    function o = o_of_beta(b)
        z = max(b - S, 0);
        if isfinite(M), z = min(z, M); end
        X = S + z;
        EX  = mean(X);
        EX2 = mean(X.^2);

        if isfinite(fmax)
            rhs = max(1/fmax, EX2/(2*b));
        else
            rhs = EX2/(2*b);
        end
        o = EX - rhs;
    end

    % bracket [l,u]
    l = 0;
    u = max(S) + 1;
    if isfinite(M), u = u + M; end
    if u <= 0, u = 1; end

    ou = o_of_beta(u);
    grow = 0;
    while ou < 0
        u = 2*u;
        ou = o_of_beta(u);
        grow = grow + 1;
        if grow > 60
            error('solve_beta_update_or_wait_crn: cannot bracket root.');
        end
    end

    % bisection
    for it = 1:200
        mid = 0.5*(l+u);
        mid = max(mid, eps);
        om  = o_of_beta(mid);
        if om >= 0
            u = mid;
        else
            l = mid;
        end
        if (u - l) <= tol
            break;
        end
    end

    beta = u;
end