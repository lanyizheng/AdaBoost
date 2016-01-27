%得到弱分类器的组合
function [weakClassArr] = adaBoostTrainDS(dataArr,classLabels,numIt)
%定义结构体变量，用来存储弱分类器
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
    %计算下一次迭代的权重D
    expon=-1*alpha.*classLabels.*bestClassEst;
    D=D.*exp(expon);
    D=D/sum(D);
    %错误率累加计算
    aggClassEst=aggClassEst+alpha*bestClassEst;
    aggErrors=(sign(aggClassEst)~=classLabels).*ones(m,1);
    errorRate=sum(aggErrors)/m;
    disp(['errorRate:',num2str(errorRate)]);
    if errorRate==0.0
        break;
    end
end
end
%生成单层决策树
%input
%-----dataArr:特征集
%-----classLabels:训练数据所对应的标签，这里以1、-1作为标签
%-----D:数据集所对应的权重
%output:
%---bestStump:在所有特征中寻找到最佳特征，以及该特征上的决策树
%---bestClassEst:使用该决策树得到的最佳结果类别
%---minError:在该决策树上最小的误差

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



