% Crop the data for testing
clc;clear;
load Label.mat label_co

label_co = label_co(:,1:18000);
save('Label.mat','label_co','-mat')
test_co = label_co(:,18001:19400);
save('ZerTest.mat','test_co','-mat')
