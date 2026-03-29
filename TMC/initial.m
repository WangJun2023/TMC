function [W, A, Z] = initial(fea, gt, anchor_num, dim_c)

view_num = length(fea);

sample_num = length(gt);

for p = 1 : view_num
    W{p} = zeros(size(fea{p}, 2), dim_c);
    A{p} = zeros(dim_c, anchor_num);
    Z{p} = zeros(anchor_num, sample_num);
    X{p} = fea{p}';
end

flag = 1;
iter = 0;
maxIter = 20;

% while flag
for iter = 1 : 20
%     iter = iter + 1;
    for p = 1 : view_num
        tmp = W{p}' * X{p} * Z{p}';
        [U,~,V] = svd(tmp, 'econ');
        A{p} = U * V';

        tmp = X{p} * Z{p}' * A{p}';
        [U,~,V] = svd(tmp, 'econ');
        W{p} = U * V';

        Z{p} = A{p}'*W{p}'*X{p};
        for ii = 1:size(Z{p},2)
            idx= 1:size(Z{p},1);
            Z{p}(idx, ii) = EProjSimplex_new(Z{p}(idx, ii));
        end
    end

%     obj_val = 0;
%     for p = 1 : view_num
%         obj_val = obj_val + trace(Z{p}'*Z{p})-2*trace(X{p}'*W{p}'*A{p}*Z{p});
%     end

%     for p = 1 : view_num
%         obj_val = obj_val + norm(W{p}*X{p}-A{p}*Z{p},'fro');
%     end

%     obj(iter) = obj_val;

%     if (iter>2) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-6 || iter>maxIter)
%         flag =0;
%     end

end


end

