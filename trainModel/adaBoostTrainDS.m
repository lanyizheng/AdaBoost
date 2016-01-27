%�õ��������������
function [weakClassArr] = adaBoostTrainDS(dataArr,classLabels,numIt)
%����ṹ������������洢��������
weakClassArr.dim=0;
weakClassArr.thresh=0;
weakClassArr.ineq=0;
weakClassArr.alpha=0;
m=size(dataArr,1);
D=ones(m,1)/m;
aggClassEst=zeros(m,1);
for i=1:numIt
    [bestStump,bestClassEst,error]=buildStump(dataArr,classLabels,D);
    %disp(['weight:',num2str(D.')]);
    alpha=0.5*log((1-error)/error);
    bestStump.alpha=alpha;
    weakClassArr(i)=bestStump;
    %disp(['classEst:',num2str(bestClassEst.')]);
    %������һ�ε�����Ȩ��D
    expon=-1*alpha.*classLabels.*bestClassEst;
    D=D.*exp(expon);
    D=D/sum(D);
    %�������ۼӼ���
    aggClassEst=aggClassEst+alpha*bestClassEst;
    aggErrors=(sign(aggClassEst)~=classLabels).*ones(m,1);
    errorRate=sum(aggErrors)/m;
    disp(['errorRate:',num2str(errorRate)]);
    if errorRate==0.0
        break;
    end
end
end
%���ɵ��������
%input
%-----dataArr:������
%-----classLabels:ѵ����������Ӧ�ı�ǩ��������1��-1��Ϊ��ǩ
%-----D:���ݼ�����Ӧ��Ȩ��
%output:
%---bestStump:������������Ѱ�ҵ�����������Լ��������ϵľ�����
%---bestClassEst:ʹ�øþ������õ�����ѽ�����
%---minError:�ڸþ���������С�����

function [bestStump,bestClassEst,minError] = buildStump(dataArr,classLabels,D)
[m,n]=size(dataArr);
numSteps=50.0;
bestStump.dim=0;
bestStump.thresh=0;
bestStump.ineq=0;
bestClassEst=zeros(m,1);
minError=inf;
Ineq=char('lt','gt');
for i=1:n
    rangeMin=min(dataArr(:,i));
    rangeMax=max(dataArr(:,i));
    stepSize=(rangeMax-rangeMin)/numSteps;
    for j=-1:(int16(numSteps)+1)
        for index=1:2
            threshVal=rangeMin+double(j)*stepSize;
            eqSign=Ineq(index,:);
            predictedVals=stumpClassify(dataArr,i,threshVal,eqSign);
            errorArr=ones(m,1);
            errorArr(predictedVals==classLabels)=0;
            weightedError=D.'*errorArr;
            if weightedError<minError
                minError=weightedError;
                bestClassEst=predictedVals;
                bestStump.dim=i;
                bestStump.thresh=threshVal;
                bestStump.ineq=eqSign;
            end
        end
    end    
end
end
function [retArray] = stumpClassify(data,dimen,threshVal,threshIneq)
retArray=ones(size(data,1),1);
if threshIneq=='lt'
    retArray(data(:,dimen)<=threshVal)=-1;
else
    retArray(data(:,dimen)>=threshVal)=-1;
end
end



