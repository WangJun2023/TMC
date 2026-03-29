function [res] = compute_singleview(Z)

[U,~,V]=svd(Z','econ');


stream = RandStream.getGlobalStream;
reset(stream);

U = U ./ repmat(sqrt(sum(U.^ 2, 2)), 1, size(U, 2));

for rep = 1 : 20
    pre = litekmeans(U, num_cluster,'MaxIter', 100,'Replicates',10); 
    res_rep(rep,:) = Clustering8Measure(gt, pre);
end
res = [mean(res_rep); std(res_rep)];
end

