function [result] =languageidentification(features)
m=size(features,1);
weakClassify=load('model.mat');
classifier=weakClassify.model;
result=adaBoostClassify(features,classifier);
end

