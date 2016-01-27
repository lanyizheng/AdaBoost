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

