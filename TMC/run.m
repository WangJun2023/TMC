% close all;
clear;
clc;
warning off;


dataPath = 'Datasets/Multi_view-Datasets/';
datasetName = {'HW', 'scene-15_3v', 'EMNIST', 'ALOI_100', 'VGGFace2_100_4Views', ...
    'MNIST_all_4views'};

ResSavePath = 'res/';
MaxResSavePath = 'maxRes/';



if(~exist(ResSavePath,'file'))
    mkdir(ResSavePath);
    addpath(genpath(ResSavePath));
end

if(~exist(MaxResSavePath,'file'))
    mkdir(MaxResSavePath);
    addpath(genpath(MaxResSavePath));
end

for dataIndex = 3
    dataName = [dataPath datasetName{dataIndex} '.mat'];
    load(dataName, 'fea', 'gt');


    fea = NormalizeData(fea);

    data = fea;

    ResBest = zeros(1, 8);
    ResStd = zeros(1, 8);
    cluster_num = length(unique(gt));

    dim_c = length(unique(gt));
    r1 = [0.00001, 0.0001, 0.001, 0.01, 0.1];
    r2 = [1*cluster_num, 2*cluster_num, 5*cluster_num];


    acc = zeros(length(r1), length(r2));
    nmi = zeros(length(r1), length(r2));
    ari = zeros(length(r1), length(r2));
    Fscore = zeros(length(r1), length(r2));

    idx = 1;
    for r2Index = 1:length(r2)
        anchor_num = r2(r2Index);
        tic;
        [W, A, Z] = initial(data, gt, anchor_num, dim_c);
        time1 = toc;
        for r1Index = 1:length(r1)
            r1Temp = r1(r1Index);
            % Main algorithm
            fprintf('Please wait a few minutes\n');
            disp(['Dataset: ', datasetName{dataIndex}, ...
                ', --r1--: ', num2str(r1Temp), ...
                ', --anchor_num--: ', num2str(anchor_num)]);
            tic;

            [res, obj1, obj2, Z_align] = main(data, gt, dim_c, r1Temp, anchor_num, W, A, Z);


            time2 = toc;
            Runtime(idx) = time1 + time2;
            fprintf('ACC=%8.6f \tNMI=%8.6f \tF1=%8.6f \tTime:%8.6f \n',[res(1, 7) res(1, 4) res(1, 1) Runtime(idx)]);
            idx = idx + 1;
            tempResBest(1, :) = res(1, :);
            tempResStd(1, :) = res(2, :);

            acc(r1Index, r2Index) = tempResBest(1, 7);
            nmi(r1Index, r2Index) = tempResBest(1, 4);
            ari(r1Index, r2Index) = tempResBest(1, 5);
            Fscore(r1Index, r2Index) = tempResBest(1, 1);

            resFile = [ResSavePath datasetName{dataIndex}, '-ACC=', num2str(tempResBest(1, 7)), ...
                '-r1=', num2str(r1Temp), ...
                '-anchor=', num2str(anchor_num), '.mat'];
            save(resFile, 'tempResBest', 'tempResStd');

            if tempResBest(1, 7) > ResBest(1, 7)
                ResBest(1, :) = tempResBest(1, :);
                ResStd(1, :) = tempResStd(1, :);
            end

        end
    end

    aRuntime = mean(Runtime);
    resFile2 = [MaxResSavePath datasetName{dataIndex}, '-ACC=', num2str(ResBest(1, 7)), '.mat'];
    save(resFile2, 'ResBest', 'ResStd', 'acc', 'nmi', 'ari', 'Fscore', 'aRuntime', 'Z_align');
end

