function [Wt,obj_old] =Train(Y, X,param)

  [n,d] = size(X);
  [~,l] = size(Y);
  Wt =ones(d,l);
  Z=eye(d,l);
  F =zeros(n,l);
  A=zeros(d,l);
  A1=zeros(n,l);
  I=eye(l,l);
  I4=eye(d,d);


param.miu=1;
param.miu1=1;
miu=param.miu;
miu_max=100;
 param.maxIter =15;
 param.tooloptions.maxiter = 15;
 param.tooloptions.gradnorm = 1e-3;
obj_old = [];
last = 0;

 
i=1;
flag=true;


%特征相似度
D=L2_distance(X,X);
S=exp(-D.^2/2);
S=S-diag(diag(S));
D=diag(S*ones(d,1));
L=D-S;
L=diag(diag(D).^(-1/2))*L*diag(diag(D).^(-1/2));
S=diag(diag(D).^(-1/2))*S*diag(diag(D).^(-1/2));

%标签相似度
S1=zeros(l,l);
for r=1:l
    for m=1:l
        if norm(Y(:,r))==0 || norm(Y(:,m))==0
           S1(r,m)=0;
        else
           S1(r,m)= Y(:,r)'*Y(:,m)/(norm(Y(:,r))*norm(Y(:,m)));
        end
    end
end
D1=diag(S1*ones(l,1));
L1=D1-S1;
%标签差异度
S_D=ones(l,l)-S1;
S_D=S_D-diag(diag(S_D));

  [~,Gw]=W_21(Wt);
while flag && i<25
   disp(i);
   disp('更新W')
   W=inv(X'*X+param.lambda1/miu*Gw+I4)*(X'*F+Z-A/miu-X'*A1/miu);
   W(W<0)=0;
  [Lw,Gw]=W_21(Z);
  Z=inv(param.lambda2/miu*Gw*S*Gw+I4)*(W+A/miu);
  Z(Z<0)=0;
  F=(Y+X*W+A1/miu)*inv(I+I+param.lambda3/miu*(L1+S_D));
obj =0.5*norm((Y-F),'fro')^2+ 0.5*norm((F-X*W),'fro')^2;
A=A+miu*(W-Z);
A(A<0)=0;
A1=A1+miu*(X*W-F);
A1(A1<0)=0;
miu=min(1.1*miu,miu_max);
convg2 = false;
    stopCriterion2 = norm(W-Wt,'fro')/norm(W,'fro');
    if stopCriterion2<1e-5
        convg2=true;
    else
        obj_old = [obj_old;stopCriterion2];
        Wt=W;
    end




          disp(obj);
          last = last + 1;
          i=i+1;

          if last < 5
              continue;
          end
          stopnow = 1;
          for ii=1:3
              stopnow = stopnow & (abs(obj-obj_old(last-1-ii))/abs(obj_old(last-1-ii)) < 1e-3);
          end
          if stopCriterion2<1e-3
              flag=false;
          end

 
 end
   
end

