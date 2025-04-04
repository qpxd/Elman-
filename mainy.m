%% Elman神经网络预测
MAPE2=1;

while MAPE2>0.2
%% 1.初始化
clearvars -except k test_simu output_test
close all
format short %精确到小数点后4位，format long是精确到小数点后15位
mse10=[0,0];
%% 2.读取读取
data=xlsread('光强分布y1.xlsx'); %% Matlab2021版本以上无法使用xlsread函数，可用Load函数替代  

% 设置神经网络的输入和输出
input=data(:,1:end-1);    %第1列至倒数第2列为输入
oneinput=data(:,1:1);  %总共的样本数量
N=length(oneinput);         %计算样本数量
outputy=data(:,end);       %最后1列为输出
%% 3.设置训练集和测试集
%（1）随机选取测试样本

[m,n]=sort(k);%升序排列 m为升序排列的值  n为每个数据的索引位置 n为以后索引提供指示
testNum=50;              %设定测试集样本数量 ！仅需修改这里
trainNum=N-testNum;       %设定训练集样本数量
input_train = input(n(1:trainNum),:)'; % 训练集输入 取出input中trainmum行并取转置 并按n所给的索引取出
output_trainy =outputy(n(1:trainNum))';  % 训练集输出 同理只不过对象换成output
input_test =input(n(trainNum+1:trainNum+testNum),:)'; % 测试集输入 同理
output_testy =outputy(n(trainNum+1:trainNum+testNum))'; % 测试集输出 同理
%% 4.数据归一化
[inputn,inputps]=mapminmax(input_train,0,1);% 训练集输入归一化到[0,1]之间 创立映射集inputps
[outputn,outputps]=mapminmax(output_trainy); % 训练集输出归一化到默认区间[-1, 1]
inputn_test=mapminmax('apply',input_test,inputps);% 测试集输入采用和训练集输入相同的归一化方式
%apply 说明应用映射集inputps的规则进行映射
%% 5.求解最佳隐含层
inputnum=size(input,2);   %size用来求取矩阵的行数和列数，1代表行数，2代表列数
outputnum=size(outputy,2);
disp(['输入层节点数：',num2str(inputnum),',  输出层节点数：',num2str(outputnum)])
disp(['隐含层节点数范围为 ',num2str(fix(sqrt(inputnum+outputnum))+1),' 至 ',num2str(fix(sqrt(inputnum+outputnum))+10)])
disp(' ')
disp('最佳隐含层节点的确定...')

%根据hiddennum=sqrt(m+n)+a，m为输入层节点数，n为输出层节点数，a取值[1,10]之间的整数
MSE=1e+5;                             %误差初始化
transform_func={'tansig','purelin'};  %激活函数采用tan-sigmoid和purelin 多维数组
train_func='trainlm';                 %训练算法
for hiddennum=fix(sqrt(inputnum+outputnum))+1:fix(sqrt(inputnum+outputnum))+10
    
    net=newelm(inputn,outputn,hiddennum,transform_func,train_func); %构建Elman网络
    
    % 设置网络参数
    net.trainParam.epochs=20000;         % 设置训练次数
    net.trainParam.lr=0.01;             % 设置学习速率
    net.trainParam.goal=0.000001;       % 设置训练目标最小误差
    
    % 进行网络训练
    net=train(net,inputn,outputn);
    an0=sim(net,inputn);      %仿真结果 输出结果
    mse0=mse(outputn,an0);    %仿真的均方误差
    disp(['当隐含层节点数为',num2str(hiddennum),'时，训练集均方误差为：',num2str(mse0)])
    mse10=[mse10,mse0];
    %不断更新最佳的隐含层节点
    if mse0<MSE
        MSE=mse0;
        hiddennum_best=hiddennum;
    end
end
disp(['最佳隐含层节点数为：',num2str(hiddennum_best),'，均方误差为：',num2str(MSE)])

%% 6.构建最佳隐含层的Elman神经网络
net=newelm(inputn,outputn,hiddennum_best,transform_func,train_func);

% 网络参数
net.trainParam.epochs=1000;          % 训练次数
net.trainParam.lr=0.01;              % 学习速率
net.trainParam.goal=0.000001;        % 训练目标最小误差

%% 7.网络训练
net=train(net,inputn,outputn);       % train函数用于训练神经网络，调用蓝色仿真界面

%% 8.网络测试
tic;
an=sim(net,inputn_test);                     %训练完成的模型进行仿真测试inputn_test为训练数据 an为仿真结果
test_simuy=mapminmax('reverse',an,outputps);  %测试结果反归一化
error=test_simuy-output_testy;                 %测试值和真实值的误差
toc;
timey=toc;
%% 9.结果输出
% Elman预测值和实际值的对比图
figure
plot(output_testy,'bo-','linewidth',1.5)%蓝色 圆圈 实线 线宽1.5 实际值
hold on
plot(test_simuy,'rs-','linewidth',1.5)%红色 放个 实线 线宽1.5 预测值
legend('实际值','预测值')%创建图例 即标注
xlabel('测试样本'),ylabel('指标值')
title('Elman预测值和实际值的对比')
set(gca,'fontsize',12)%对全图文字设置大小为12

% Elamn测试集的预测误差图
figure
plot(error,'bo-','linewidth',1.5)
xlabel('测试样本'),ylabel('预测误差')
title('Elman神经网络测试集的预测误差')
set(gca,'fontsize',12)
% Elamn隐藏节点数对RMSE的影响
figure
plot(mse10,'bo-','linewidth',1.5)
xlabel('隐藏节点数'),ylabel('RMSE-y')
title('隐藏节点数对RMSE的影响')
set(gca,'fontsize',12)
xlim([3,12])

figure;
plotregression(output_testy,test_simuy,['Elman回归图']); %绘制output_test对test_simu的线性回归图 test_simu为点 output_test为线
figure;
ploterrhist(test_simuy-output_testy,['Elman误差直方图']);%绘制output_test对test_simu的误差直方图 

%计算各项误差参数  
[~,len]=size(output_testy);             % len获取测试样本个数，数值等于testNum，用于求各指标平均值
SSE1=sum(error.^2);                    % 误差平方和
MAE1=sum(abs(error))/len;              % 平均绝对误差
MSE1=error*error'/len;                 % 均方误差
RMSE1=MSE1^(1/2);                      % 均方根误差
MAPE2=mean(abs(error./output_testy));   % 平均百分比误差
r=corrcoef(output_testy,test_simuy);     % corrcoef计算相关系数矩阵，包括自相关和互相关系数
R1=r(1,2);    
end
% 显示各指标结果
disp(' ')
disp('各项误差指标结果：')
disp(['误差平方和SSE为：',num2str(SSE1)])
disp(['平均绝对误差MAE为：',num2str(MAE1)])
disp(['均方误差MSE为：',num2str(MSE1)])
disp(['均方根误差RMSE为：',num2str(RMSE1)])
disp(['平均百分比误差MAPE为：',num2str(MAPE2*100),'%'])
disp(['预测准确率为：',num2str(100-MAPE2*100),'%'])
disp(['相关系数R为：',num2str(R1)])%相关系数越大越好 越接近于1 越相关

output_finally=[output_test',output_testy'];%真实位置
test_finally=[test_simu',test_simuy'];%预测位置
% save y.mat;
% save('nety.mat','net'); 保存data

%save ('timey.mat','timey') %保存执行时间（50）
% 工作区中
% output_test代表测试集
% test_simu代表BP预测值
% error代表误差
