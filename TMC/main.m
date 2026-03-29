function [res, obj1, obj2, Z_align] = main(fea, gt, dim_c, r1Temp, anchor_num, W, A, Z)

num_cluster = length(unique(gt));
view_num = length(fea);
sample_num = length(gt);

for v = 1 : view_num
    X{v} = fea{v}';

    G{v} = zeros(dim_c, anchor_num);
    J{v} = zeros(dim_c, anchor_num);

    H{v} = zeros(anchor_num, sample_num);
    M{v} = zeros(anchor_num, sample_num);
end

sg = [dim_c, anchor_num, view_num];
sh = [anchor_num, sample_num, view_num];

mu = 1e-5;
max_mu = 10e10;
rho = 1e-3;
max_rho = 10e12;
eta = 2;
lambda = r1Temp;

flag = 1;
iter = 0;
maxiter = 50;
time_w_sum = 0;
time_z_sum = 0;
time_a_sum = 0;
time_g_sum = 0;
time_h_sum = 0;
time_j_sum = 0;
time_m_sum = 0;
time_mu_sum = 0;
time_rho_sum = 0;

while flag
    iter = iter + 1;

    for p = 1 : view_num
        %% Udpate W
        tic
        tmp = lambda * X{p} * Z{p}' * A{p}';
        [U,~,V] = svd(tmp, 'econ');
        W{p} = U * V';
        time_W = toc;
        time_w_sum = time_w_sum + time_W;

        %% Update Z
        tic
        Z{p} = 1/(2+rho)*(rho*H{p}+2*lambda*A{p}'*W{p}'*X{p}-M{p});
        for ii = 1:size(Z{p},2)
            idx= 1:size(Z{p},1);
            Z{p}(idx, ii) = EProjSimplex_new(Z{p}(idx, ii));
        end
        time_Z = toc;
        time_z_sum = time_z_sum + time_Z;

        %% Update A
        tic
        tmp = J{p} - mu*G{p} - 2*lambda*W{p}'*X{p}*Z{p}';
        [U,~,V] = svd(-tmp, 'econ');
        A{p} = U * V';
        time_A = toc;
        time_a_sum = time_a_sum + time_A;
    end

    %% Update G
    tic
    A_tensor = cat(3, A{:,:});
    J_tensor = cat(3, J{:,:});
    a = A_tensor(:);
    j = J_tensor(:);

    [g, obj_g] = wshrinkObj(a + 1/mu*j,1/mu,sg,0,3);
    G_tensor = reshape(g, sg);
    time_G = toc;
    time_g_sum = time_g_sum + time_G;

    %% Update H
    tic
    Z_tensor = cat(3, Z{:,:});
    M_tensor = cat(3, M{:,:});
    z = Z_tensor(:);
    m = M_tensor(:);

    [h, obj_h] = wshrinkObj(z + 1/rho*m,1/rho,sh,0,3);
    H_tensor = reshape(h, sh);
    time_H = toc;
    time_h_sum = time_h_sum + time_H;

    %% Update J
    tic
    j = j + mu*(a-g);
    time_j = toc;
    time_j_sum = time_j_sum + time_j;

    %% Update M
    tic
    m = m + rho*(z-h);
    time_m = toc;
    time_m_sum = time_m_sum + time_m;

    %% Update mu
    tic
    mu = min(mu*eta, max_mu);
    time_mu = toc;
    time_mu_sum = time_mu_sum + time_mu;

    %% Update rho
    tic
    rho = min(rho*eta, max_rho);
    time_rho = toc;
    time_rho_sum = time_rho_sum + time_rho;

    %% Converge Condition
    flag = 0;
    for p = 1 : view_num
        G{p} = G_tensor(:,:,p);
        J_tensor = reshape(j, sg);
        J{p} = J_tensor(:,:,p);
        if (norm(A{p}-G{p}, inf) > 1e-7)
            flag = 1;
            obj1(iter) = norm(A{p}-G{p}, inf);
        end

        H{p} = H_tensor(:,:,p);
        M_tensor = reshape(m, sh);
        M{p} = M_tensor(:,:,p);
        if (norm(Z{p}-H{p}, inf) > 1e-7)
            flag = 1;
            obj2(iter) = norm(Z{p}-H{p}, inf);
        end
    end

    if iter > maxiter
        flag = 0;
    end

end

Z_align = zeros(anchor_num, sample_num);
for p = 1 : view_num
    Z_align = Z_align + Z{p};
end


[U,~,V] = svd(Z_align','econ');

stream = RandStream.getGlobalStream;
reset(stream);

U = U ./ repmat(sqrt(sum(U.^ 2, 2)), 1, size(U, 2));

for rep = 1 : 20
    pre = litekmeans(U, num_cluster,'MaxIter', 100,'Replicates',10);
    res_rep(rep,:) = Clustering8Measure(gt, pre);
end
res = [mean(res_rep); std(res_rep)];

end

