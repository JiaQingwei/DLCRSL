function [in_result, out_result,Out_parameter] = run_arts()


%param = importdata('arts_param.mat');
data = importdata('BUS.mat');
N=floor(1*floor(1*size(data.features,1)));
Ntrain=floor(0.6*N);
param.tooloptions.maxiter = 15;
param.tooloptions.gradnorm = 1e-3;
param.tooloptions.stopfun = @mystopfun;
 Lambda5=[0.001,0.01,0.02,0.03,0.05,0.08,0.11,0.15,0.20,0.25,0.35,0.55,0.75,1];

  param.lambda = 0;
  param.lambda1 =10;%ȫ�ֱ�ǩ������ƶȾ���L1
  param.lambda2 =1;%L�ĸ�������Q��������
  param.lambda3 =0.1;%V
  param.lambda4 =0;%����������
  param.lambda5 =0;%W,U,C,Z,Q 
  param.lambda6 = 0;%P��Lx��F����������
param.lambda7 =0;%C������ϵ��
param.lambda8=0;%P��������
 out_result = [];
 in_result = [];
 Out_parameter=[];
%��ԭ���������е�û�е���������0��ȫ���ã�-1����ʾ
%data.features(data.features==0)=-1;
%data.features=mapminmax(data.features,0,1);
%  %��ԭ����ǩ���е�û�еı�ǩ����0��ȫ���ã�-1����ʾ
 data.labels(data.labels==-1)=0;
  

for j=1:5
    for kk=1:5
indices = crossvalind('Kfold', 1:N ,5);    
test_idxs = (indices == kk);
train_idxs = ~test_idxs;
 Xtrn = data.features(train_idxs,:);
 Ytrn = data.labels(train_idxs,:);
 Xtst = data.features(test_idxs,:);
 Ytst = data.labels(test_idxs,:);
 Xtrn=mapminmax(Xtrn,0,1);
 Xtst=mapminmax(Xtst,0,1);
            tic;
            zz = mean(Ytrn,2);
            Ytrn(zz==0,:) = [];
            Xtrn(zz==0,:) = [];

            [W,obj_old] = Train(Ytrn, Xtrn,param);
            tm = toc;
            zz = mean(Ytst,2);
            Ytst(zz==0,:) = [];
            Xtst(zz==0,:) = [];
            tstv = (Xtst*W);
            ret =  evalt(tstv',Ytst', (max(tstv(:))-min(tstv(:)))/2);
           out_result=[out_result;ret];

        for i=1:1  
            NUM=10;
            [Result_LRMLFSl] = feature_selection(W,Xtrn,Xtst,Ytrn,Ytst,NUM);
           RESULT=mean(Result_LRMLFSl(1:5,:));
          Out_parameter=[Out_parameter;RESULT];

       end
    end
 end

 end


function stopnow = mystopfun(problem, x, info, last)
    if last < 5 
        stopnow = 0;
        return;
    end
    flag = 1;
    for i = 1:3
        flag = flag & abs(info(last-i).cost-info(last-i-1).cost) < 1e-1;
    end
    stopnow = flag;
end