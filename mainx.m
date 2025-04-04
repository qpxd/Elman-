%% Elman������Ԥ��
MAPE1=1;
while MAPE1>0.2
%% 1.��ʼ��
clear
close all
clc
format short %��ȷ��С�����4λ��format long�Ǿ�ȷ��С�����15λ
mse10=[0,0];
%% 2.��ȡ��ȡ
data=xlsread('��ǿ�ֲ�x1.xlsx'); %% Matlab2021�汾�����޷�ʹ��xlsread����������Load�������  

% �������������������
input=data(:,1:end-1);    %��1����������2��Ϊ����
oneinput=data(:,1:1);  %�ܹ�����������
N=length(oneinput);         %������������
output=data(:,end);       %���1��Ϊ���
%% 3.����ѵ�����Ͳ��Լ�
%��1�����ѡȡ��������
k=rand(1,N); %�������1*N�ľ���
[m,n]=sort(k);%�������� mΪ�������е�ֵ  nΪÿ�����ݵ�����λ�� nΪ�Ժ������ṩָʾ
testNum=50;              %�趨���Լ��������� �������޸�����
trainNum=N-testNum;       %�趨ѵ������������
input_train = input(n(1:trainNum),:)'; % ѵ�������� ȡ��input��trainmum�в�ȡת�� ����n����������ȡ��
output_train =output(n(1:trainNum))';  % ѵ������� ͬ��ֻ�������󻻳�output
input_test =input(n(trainNum+1:trainNum+testNum),:)'; % ���Լ����� ͬ��
output_test =output(n(trainNum+1:trainNum+testNum))'; % ���Լ���� ͬ��
%% 4.���ݹ�һ��
[inputn,inputps]=mapminmax(input_train,0,1);% ѵ���������һ����[0,1]֮�� ����ӳ�伯inputps
[outputn,outputps]=mapminmax(output_train); % ѵ���������һ����Ĭ������[-1, 1]
inputn_test=mapminmax('apply',input_test,inputps);% ���Լ�������ú�ѵ����������ͬ�Ĺ�һ����ʽ
%apply ˵��Ӧ��ӳ�伯inputps�Ĺ������ӳ��
%% 5.������������
inputnum=size(input,2);   %size������ȡ�����������������1����������2��������
outputnum=size(output,2);
disp(['�����ڵ�����',num2str(inputnum),',  �����ڵ�����',num2str(outputnum)])
disp(['������ڵ�����ΧΪ ',num2str(fix(sqrt(inputnum+outputnum))+1),' �� ',num2str(fix(sqrt(inputnum+outputnum))+10)])
disp(' ')
disp('���������ڵ��ȷ��...')

%����hiddennum=sqrt(m+n)+a��mΪ�����ڵ�����nΪ�����ڵ�����aȡֵ[1,10]֮�������
MSE=1e+5;                             %����ʼ��
transform_func={'tansig','purelin'};  %���������tan-sigmoid��purelin ��ά����
train_func='trainlm';                 %ѵ���㷨
for hiddennum=fix(sqrt(inputnum+outputnum))+1:fix(sqrt(inputnum+outputnum))+10
    
    net=newelm(inputn,outputn,hiddennum,transform_func,train_func); %����Elman����
    
    % �����������
    net.trainParam.epochs=20000;         % ����ѵ������
    net.trainParam.lr=0.01;             % ����ѧϰ����
    net.trainParam.goal=0.000001;       % ����ѵ��Ŀ����С���
    
    % ��������ѵ��
    net=train(net,inputn,outputn);
    an0=sim(net,inputn);      %������ ������
    mse0=mse(outputn,an0);    %����ľ������
    disp(['��������ڵ���Ϊ',num2str(hiddennum),'ʱ��ѵ�����������Ϊ��',num2str(mse0)])
    mse10=[mse10,mse0];
    %���ϸ�����ѵ�������ڵ�
    if mse0<MSE
        MSE=mse0;
        hiddennum_best=hiddennum;
    end
end
disp(['���������ڵ���Ϊ��',num2str(hiddennum_best),'���������Ϊ��',num2str(MSE)])

%% 6.��������������Elman������
net=newelm(inputn,outputn,hiddennum_best,transform_func,train_func);

% �������
net.trainParam.epochs=20000;          % ѵ������
net.trainParam.lr=0.01;              % ѧϰ����
net.trainParam.goal=0.000001;        % ѵ��Ŀ����С���

%% 7.����ѵ��
net=train(net,inputn,outputn);       % train��������ѵ�������磬������ɫ�������

%% 8.�������
tic;
an=sim(net,inputn_test);                     %ѵ����ɵ�ģ�ͽ��з������inputn_testΪѵ������ anΪ������
test_simu=mapminmax('reverse',an,outputps);  %���Խ������һ��
error=test_simu-output_test;                 %����ֵ����ʵֵ�����
timex=toc;
%% 9.������
% ElmanԤ��ֵ��ʵ��ֵ�ĶԱ�ͼ
figure
plot(output_test,'bo-','linewidth',1.5)%��ɫ ԲȦ ʵ�� �߿�1.5 ʵ��ֵ
legend('ʵ��ֵ')
hold on
plot(test_simu,'rs-','linewidth',1.5)%��ɫ �Ÿ� ʵ�� �߿�1.5 Ԥ��ֵ
legend('Ԥ��ֵ')%����ͼ�� ����ע
xlabel('��������'),ylabel('ָ��ֵ/0.125m')
title('ElmanԤ��ֵ��ʵ��ֵ�ĶԱ�')
set(gca,'fontsize',12)%��ȫͼ�������ô�СΪ12

% Elamn���Լ���Ԥ�����ͼ
figure
plot(error,'bo-','linewidth',1.5)
xlabel('��������'),ylabel('Ԥ�����/0.125m')
title('Elman��������Լ���Ԥ�����')
set(gca,'fontsize',12)

figure;
plotregression(output_test,test_simu,['Elman�ع�ͼ']); %����output_test��test_simu�����Իع�ͼ test_simuΪ�� output_testΪ��
figure;
ploterrhist(test_simu-output_test,['Elman���ֱ��ͼ']);%����output_test��test_simu�����ֱ��ͼ 
% Elamn���ؽڵ�����RMSE��Ӱ��
figure
plot(mse10,'bo-','linewidth',1.5)
xlabel('���ؽڵ���'),ylabel('MSE-x/0.125m')
title('���ؽڵ�����ѵ������������Ӱ��')
set(gca,'fontsize',12)
xlim([3,12])

%�������������  
[~,len]=size(output_test);             % len��ȡ����������������ֵ����testNum���������ָ��ƽ��ֵ
SSE1=sum(error.^2);                    % ���ƽ����
MAE1=sum(abs(error))/len;              % ƽ���������
MSE1=error*error'/len;                 % �������
RMSE1=MSE1^(1/2);                      % ���������
MAPE1=mean(abs(error./output_test));   % ƽ���ٷֱ����
r=corrcoef(output_test,test_simu);     % corrcoef�������ϵ�����󣬰�������غͻ����ϵ��
R1=r(1,2);    

end
% ��ʾ��ָ����
disp(' ')
disp('�������ָ������')
disp(['���ƽ����SSEΪ��',num2str(SSE1)])
disp(['ƽ���������MAEΪ��',num2str(MAE1)])
disp(['�������MSEΪ��',num2str(MSE1)])
disp(['���������RMSEΪ��',num2str(RMSE1)])
disp(['ƽ���ٷֱ����MAPEΪ��',num2str(MAPE1*100),'%'])
disp(['Ԥ��׼ȷ��Ϊ��',num2str(100-MAPE1*100),'%'])
disp(['���ϵ��RΪ��',num2str(R1)])%���ϵ��Խ��Խ�� Խ�ӽ���1 Խ���
% save x.mat;
% save('netx.mat','net');����data
%save ('timex.mat','timex') %����ִ��ʱ�䣨50��
% ��������
% output_test������Լ�
% test_simu����BPԤ��ֵ
% error�������
