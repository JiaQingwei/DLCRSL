function [Result_LRMLFSl] = feature_selection(W,Xtrn,Xtst,Ytrn,Ytst,NUM)
% W=W';
% Xtrn=Xtrn';
% Xtst=Xtst';
% Ytrn=Ytrn';
% Ytst=Ytst';
% clc;
% addpath('clf');
% addpath('lib');
% addpath(genpath('lib/manopt'));
% addpath('evl');
% 
% data = importdata('dt/Education_5000_550_33.mat');
% Xtrn = data.train{1,1};
% Xtrn = normlize_data(Xtrn');   %%  n*d
% Xtrn=Xtrn';
% Ytrn = data.train{1,2};   %%  n*c
% aa=[  0.00001 0.0001 0.001 0.01 0.1 1 2 3 4 5 6 7 8 9 10 20 15 35 45 50 60];

% for kk=1:length(aa)
%     clc;
% addpath('clf');
% addpath('lib');
% addpath(genpath('lib/manopt'));
% addpath('evl');
% 
% data = importdata('dt/Image_2000_294_5.mat');
% Xtrn = data.train{1,1};
% Xtrn = normlize_data(Xtrn');   %%  n*d 
% Xtrn=Xtrn';
% Ytrn = data.train{1,2};   %%  n*c
% % a=aa(kk);
% ee = 0.01;
% a=0.07;
%[ W ] = LRMLFS( Xtrn',Ytrn,ee,a );

% Xtest = data.test{1,1};
% Xtest = normlize_data(Xtest');   %%  n*d
% Ytest = data.test{1,2};   %%  n*c

num_feature = size(Xtrn,2);
%[dumb idx] = sort(sum(W.*W,2),'descend');

%% The default setting of MLKNN
Num =NUM;Smooth = 1;
% Xtrn=Xtrn';
% Xtest=Xtest';
% cv_train_data = data.train{1,1}';
% cv_train_target = data.train{1,2};
% cv_test_data = data.test{1,1}';
% cv_test_target = data.test{1,2};
cv_train_data = Xtrn;
cv_train_target = Ytrn;
cv_test_data = Xtst;
cv_test_target = Ytst;
tmp_cv_train_target = cv_train_target';
%tmp_cv_train_target(tmp_cv_train_target==0) = -1;
tmp_cv_test_target = cv_test_target';
%tmp_cv_test_target(tmp_cv_test_target==0) = -1;
Result_LRMLFSl=zeros(10,1);

[dumb , idx] = sort(sum(W.*W,2),'descend');
selectedFN =floor( 0.1*num_feature);
 for FeaNum=selectedFN:selectedFN:num_feature
% for FeaNum=70:10:10*selectedFN
    fea = idx(1:FeaNum);
    % ML-KNN classifier
    [Prior,PriorN,Cond,CondN]=MLKNN_train(cv_train_data(:,fea),tmp_cv_train_target,Num,Smooth);
    [HammingLoss,RankingLoss,Coverage,OneError,Average_Precision,AUC,macrof1,microf1,EBA,EBP,EBR,EBF,LBA,LBP,LBR,LBF]= MLKNN_test(cv_train_data(:,fea),tmp_cv_train_target,cv_test_data(:,fea),tmp_cv_test_target,Num,Prior,PriorN,Cond,CondN);
    Result_LRMLFSl(FeaNum/selectedFN,1)=HammingLoss;
    Result_LRMLFSl(FeaNum/selectedFN,2)=RankingLoss;
    Result_LRMLFSl(FeaNum/selectedFN,3)=Coverage;
    Result_LRMLFSl(FeaNum/selectedFN,4)=OneError;
    Result_LRMLFSl(FeaNum/selectedFN,5)=Average_Precision;
    Result_LRMLFSl(FeaNum/selectedFN,6)=AUC;
    Result_LRMLFSl(FeaNum/selectedFN,7)=macrof1;
    Result_LRMLFSl(FeaNum/selectedFN,8)=microf1;
    
end

%Result_LRMLFSl;
% subplot(2,3,1)
% plot(HL_MDFS)
% subplot(2,3,2)
% plot(RL_MDFS)
% subplot(2,3,3)
% plot(CV_MDFS)
% subplot(2,3,4)
% plot(AP_MDFS)
% subplot(2,3,5)
% plot(MA_MDFS)
% subplot(2,3,6)
% plot(MI_MDFS)

% HH{1,kk}=Result_LRMLFSl;
% end



