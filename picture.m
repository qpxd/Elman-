% 预测整体误差
figure
plot(p_true(:,1),'bs-','linewidth',1.5)%蓝色 圆圈 实线 线宽1.5 实际值
hold on
plot(p_true(:,2),'bo-','linewidth',1.5)%蓝色 圆圈 实线 线宽1.5 实际值
hold on
plot(p_test(:,1),'rs-','linewidth',1.5)%蓝色 圆圈 实线 线宽1.5 实际值
hold on
plot(p_test(:,2),'ro-','linewidth',1.5)%红色 放个 实线 线宽1.5 预测值
legend('实际值-x','实际值-y','KNN预测值-x','KNN预测值-y')%创建图例 即标注
xlabel('测试样本'),ylabel('指标值/0.125m')
title('KNN修正预测值和实际值的对比')
set(gca,'fontsize',12)%对全图文字设置大小为12
%神经网络误差对比图

figure
c = categorical({'Elman-x','Elman-y','Elman-mean','Elman-finally',});
% x = [1,2,3,4];
y = [2.2447,2.0266,2.13565,2.0687];
bar(c,y);
title('不同神经网络均方根误差对比');
ylabel('指标值/0.125m');

figure
c = categorical({'Elman-x','Elman-y','Elman-mean','Elman-finally','Elamn-KNN'});
% x = [1,2,3,4];
y = [2.2447,2.0266,2.13565,2.0687,error_K(2)];
bar(c,y,'m');
title('不同神经网络均方根误差对比（总）');
ylabel('指标值/0.125m');

