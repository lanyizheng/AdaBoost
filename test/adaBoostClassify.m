%通过已知模型，根据特征预测标签，这里讲1、-1转换成1、2
function [classLabels] = adaBoostClassify(dataMat,classifier)
m=size(dataMat,1);
classLabels=zeros(m,1);
n=size(classifier,2);
for i=1:n
    classEst=stumpClassify(dataMat,classifier(i).dim,classifier(i).thresh,classifier(i).ineq);
    classLabels=classLabels+classifier(i).alpha*classEst;
    %disp([num2str(classLabels.')]);
end
classLabels=sign(classLabels);
classLabels(classLabels==-1*ones(m,1))=2;
end
function [retArray] = stumpClassify(data,dimen,threshVal,threshIneq)
retArray=ones(size(data,1),1);
if threshIneq=='lt'
    retArray(data(:,dimen)<=threshVal)=-1;
else
    retArray(data(:,dimen)>=threshVal)=-1;
end
end

