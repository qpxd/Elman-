%% 数据说明
%fingerprint_sim:   指纹数据库,21m*21m, 4AP
%RSS_fp:    50组测试数据的RSS
%p_true:    50组测试数据的真实位置
%p_test     50组测试数据的预测位置
fingerprint_sim=xlsread('光强分布总1.xlsx');
FP1=fingerprint_sim(:,1);
FP1=reshape(FP1,[21,21]);
FP2=fingerprint_sim(:,2);
FP2=reshape(FP2,[21,21]);
FP3=fingerprint_sim(:,3);
FP3=reshape(FP3,[21,21]);
FP4=fingerprint_sim(:,4);
FP4=reshape(FP4,[21,21]);
fingerprint_sim=cat(4,FP1,FP2,FP3,FP4);%指纹数据库
p_true=output_finally;% 真实位置
RSS_fp=input_test'; %预测位置的指纹数据
p_test=test_finally; % 预测位置
p_KNN = 0;%存定位结果
hengzuobiao=[0,1,2,3,4,5,6,7,8,9,10];
%% 判断矩阵
n = 4;%使用AP的数目，这里使用全部6个（<=n_AP）
error_test=mean(sqrt((p_true(:,1)-p_test(:,1)).^2+(p_true(:,2)-p_test(:,2)).^2));%未修正前平均误差
error_K=[error_test];%初始误差
for k = 1:10%KNN算法中的K
for i=1:size(p_test)
    error_every=sqrt((p_true(i,1)-p_test(i,1))^2+(p_true(i,2)-p_test(i,2))^2);
    if error_every>error_test
        
    %按顺序分别给每一个数据定位
        [size_x, size_y, n_AP] = size(fingerprint_sim);
    %计算欧氏距离
        distance = 0;
        for j=1:n
            distance = distance + (fingerprint_sim(:,:,j)-RSS_fp(i,j)).^2;%这里同时计算所有参考点，结果是一个二维矩阵
        end
        distance = sqrt(distance);
    %将欧氏距离排序，选择k个最小的，得到位置
        d = reshape(distance,1,size_x*size_y);
        [whatever, index_d]=sort(d);%排序
        knn_x = (mod(index_d(1:k),size_x));%取余数
        knn_y = (floor(index_d(1:k)./size_x)+1);%取整函数
        p_KNN(i,1:2) = [mean(knn_x), mean(knn_y)];%k个位置求平均
    else
        p_KNN(i,1:2)=p_test(i,1:2);
    end
end
error_KNN=sqrt((p_true(:,1)-p_KNN(:,1)).^2+(p_true(:,2)-p_KNN(:,2)).^2);
disp('KNN平均误差：')
disp(mean(error_KNN));
error_K=[error_K,mean(error_KNN)];
end
% Elamn隐藏节点数对RMSE的影响
figure
plot(hengzuobiao, error_K,'bo-','linewidth',1.5)
xlabel('k的值'),ylabel('RMSE-k/0.125m')
title('k值选取对RMSE的影响')
set(gca,'fontsize',12)
% save k.mat; 保存data