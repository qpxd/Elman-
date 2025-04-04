%% ����˵��
%fingerprint_sim:   ָ�����ݿ�,21m*21m, 4AP
%RSS_fp:    50��������ݵ�RSS
%p_true:    50��������ݵ���ʵλ��
%p_test     50��������ݵ�Ԥ��λ��
fingerprint_sim=xlsread('��ǿ�ֲ���1.xlsx');
FP1=fingerprint_sim(:,1);
FP1=reshape(FP1,[21,21]);
FP2=fingerprint_sim(:,2);
FP2=reshape(FP2,[21,21]);
FP3=fingerprint_sim(:,3);
FP3=reshape(FP3,[21,21]);
FP4=fingerprint_sim(:,4);
FP4=reshape(FP4,[21,21]);
fingerprint_sim=cat(4,FP1,FP2,FP3,FP4);%ָ�����ݿ�
p_true=output_finally;% ��ʵλ��
RSS_fp=input_test'; %Ԥ��λ�õ�ָ������
p_test=test_finally; % Ԥ��λ��
p_KNN = 0;%�涨λ���
hengzuobiao=[0,1,2,3,4,5,6,7,8,9,10];
%% �жϾ���
n = 4;%ʹ��AP����Ŀ������ʹ��ȫ��6����<=n_AP��
error_test=mean(sqrt((p_true(:,1)-p_test(:,1)).^2+(p_true(:,2)-p_test(:,2)).^2));%δ����ǰƽ�����
error_K=[error_test];%��ʼ���
for k = 1:10%KNN�㷨�е�K
for i=1:size(p_test)
    error_every=sqrt((p_true(i,1)-p_test(i,1))^2+(p_true(i,2)-p_test(i,2))^2);
    if error_every>error_test
        
    %��˳��ֱ��ÿһ�����ݶ�λ
        [size_x, size_y, n_AP] = size(fingerprint_sim);
    %����ŷ�Ͼ���
        distance = 0;
        for j=1:n
            distance = distance + (fingerprint_sim(:,:,j)-RSS_fp(i,j)).^2;%����ͬʱ�������вο��㣬�����һ����ά����
        end
        distance = sqrt(distance);
    %��ŷ�Ͼ�������ѡ��k����С�ģ��õ�λ��
        d = reshape(distance,1,size_x*size_y);
        [whatever, index_d]=sort(d);%����
        knn_x = (mod(index_d(1:k),size_x));%ȡ����
        knn_y = (floor(index_d(1:k)./size_x)+1);%ȡ������
        p_KNN(i,1:2) = [mean(knn_x), mean(knn_y)];%k��λ����ƽ��
    else
        p_KNN(i,1:2)=p_test(i,1:2);
    end
end
error_KNN=sqrt((p_true(:,1)-p_KNN(:,1)).^2+(p_true(:,2)-p_KNN(:,2)).^2);
disp('KNNƽ����')
disp(mean(error_KNN));
error_K=[error_K,mean(error_KNN)];
end
% Elamn���ؽڵ�����RMSE��Ӱ��
figure
plot(hengzuobiao, error_K,'bo-','linewidth',1.5)
xlabel('k��ֵ'),ylabel('RMSE-k/0.125m')
title('kֵѡȡ��RMSE��Ӱ��')
set(gca,'fontsize',12)
% save k.mat; ����data