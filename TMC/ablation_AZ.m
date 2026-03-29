function [res, Z_align] = ablation_AZ(fea, gt, anchor_num, dim_c)

num_cluster = length(unique(gt));
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
maxIter = 50;

while flag
    iter = iter + 1;

    %% Udpate W
    for p = 1 : view_num 
        tmp = X{p} * Z{p}' * A{p}';
        [U,~,V] = svd(tmp, 'econ');
        W{p} = U * V';

    %% Update Z
        Z{p} = (A{p}'*W{p}'*X{p});
%         for ii = 1:size(Z{p},2)
%             idx= 1:size(Z{p},1);
%             Z{p}(idx, ii) = EProjSimplex_new(Z{p}(idx, ii));
%         end

    %% Update A
        tmp = W{p}'*X{p}*Z{p}';
        [U,~,V] = svd(tmp, 'econ');
        A{p} = U * V';
    end
    
    obj_value = 0;
    for p = 1 : view_num
        obj_value = obj_value + norm(X{p}-W{p}*A{p}*Z{p}, 'fro')^2;
    end
   obj(iter) =  obj_value;
% 
    if (iter>2) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-6 || iter>maxIter)
        flag =0;
    end

%     if iter > maxIter
%         flag = 0;
%     end


end

Z_align = zeros(anchor_num, sample_num);
for p = 1 : view_num
    Z_align = Z_align + Z{p}; 
end

% imagesc(Z_align' * Z_align);
% colorbar;
% PicPath = ['C:/Users/Administrator/Desktop/pic/image/' num2str(idx),'.jpg'];
% print(gcf, '-djpeg','-r300', PicPath);
% close(gcf);

[U,~,V]=svd(Z_align','econ');


stream = RandStream.getGlobalStream;
reset(stream);

U = U ./ repmat(sqrt(sum(U.^ 2, 2)), 1, size(U, 2));

for rep = 1 : 20
    pre = litekmeans(U, num_cluster,'MaxIter', 100,'Replicates',10); 
    res_rep(rep,:) = Clustering8Measure(gt, pre);
end
res = [mean(res_rep); std(res_rep)];

end

