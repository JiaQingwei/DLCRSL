function [W_21,Grad_W21]=W_21(W)
[d,m]=size(W);
W_21=0;
Grad_W21=zeros(d,d);
for i=1:d
    term=0;
    for j=1:m
     term= term +W(i,j)^2;
    end
    W_21=W_21+sqrt(term);
%     if term ==0
%     Grad_W21(i,i)=1/(eps);
%     else
      Grad_W21(i,i)=1/(sqrt(term)+0.0001);  
    %end
end

