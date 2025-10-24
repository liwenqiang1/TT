import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

#sys.path.append(r'C:\Users\wlj13\PycharmProjects\pythonProject')
#import os

#os.environ['PATH']
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#file1="2022_11_01-30.csv"
file1 = "D:\\qingxi\\data\\2022_11_01-30.csv"
OriginData = pd.read_csv(file1)
OriginData['created_at'] = pd.to_datetime(OriginData['created_at'])
OriginData.set_index('created_at',drop=True, append=False, inplace=True)
QuarterData = OriginData.groupby(pd.Grouper(freq="15T")).mean()
QuarterData = QuarterData.reset_index()

QuarterData.rename(columns={
    "AI_P1_INCOME_SH_PH": "PH",
    "AI_P1_INCOME_SH_TN":"TN",
    "AI_P1_INCOME_SH_COD":"COD",
    "PLC1_CCC_LL_FLOW":"FLOW",
    "SQP2_SHC_WaterTemp":"WATERTEMP",
    "SQP2_SHC_ORP":"ORP1",
    "SQP2_SHC_ORP_L2":"ORP2",
    "PLC2_SHC_B_SS1":"SS1",
    "PLC2_SHC_B_SS2":"SS2",
    "PLC2_WNBF_WHL_LL":"WHL",
    "PLC3_YSN_LL_FLOW2":"YSN",
    "PLC2_GFJF_LL_FLOW":"GFJF",
    "PLC2_SHC_B_RJY1": "RJY1",
    "PLC2_SHC_B_RJY2": "RJY2" },inplace=True)
#print(QuarterData.shape)
data = QuarterData[['FLOW','GFJF','RJY1','RJY2']]
#data=QuarterData.loc[(QuarterData['TN']<=40) & (QuarterData['COD']<=400),['FLOW','GFJF','RJY1','RJY2']]

#data=QuarterData.loc[(QuarterData['TN']<=40) & (QuarterData['COD']<=400) & (QuarterData['GFJF']>=1500) & (QuarterData['GFJF']<=3500) & (QuarterData['RJY1']<=4) & (QuarterData['RJY1']>=1.5) & (QuarterData['RJY2']>=0.3) & (QuarterData['RJY2']<=3),['FLOW','GFJF','RJY1','RJY2']]
#data.to_csv('k_regressor.csv')
#print(data.shape)

#以下全都是回归预测

FLOW = data['FLOW'].values
delta_V = [i*0.25*0.8 for i in FLOW]
dV = (28.8*8*6+28.8*4.5*6)*0.25/(3/4)

GFJF = data['GFJF'].values
#Q=[i for i in GFJF]
Q = [i*1.43*1000*0.21*0.25*(2/44) for i in GFJF]

real_gfjf=[i for i in GFJF]
real_gfjf.pop()

RJY1 = data['RJY1'].values
RJY2 = data['RJY2'].values

#C_times_V=[RJY1[i]*(1206*6+1382.4)+RJY2[i]*777.6 for i in range(len(RJY1))]
#C_times_V=[RJY1[i]*1382.4+RJY2[i]*777.6 for i in range(len(RJY1))]
C_times_V = [Q[i]+RJY2[i]*777.6 for i in range(len(RJY1))]

CV_diff = []
for i in range(len(C_times_V)-1):
    CV_diff.append(C_times_V[i+1]-C_times_V[i])
C_times_V.pop()
Q.pop()

C2_deltaV=[]
C2_RJY2=[]
for i in range(len(RJY2)):
    C2_deltaV.append(RJY2[i] * dV)
    C2_RJY2.append(RJY2[i])
C2_deltaV.pop()
C2_RJY2.pop()
C1_deltaV=[]
for i in range(len(RJY1)):
    #C2_deltaV.append(RJY1[i]*delta_V[2])
    C1_deltaV.append(RJY1[i] * dV)
C1_deltaV.pop()

test_label = int(0.2*len(C_times_V))
Q_test = Q[:test_label]
Q_train = Q[test_label:]
C1_deltaV_test = C1_deltaV[:test_label]
C1_deltaV_train = C1_deltaV[test_label:]
C2_deltaV_test = C2_deltaV[:test_label]
C2_deltaV_train = C2_deltaV[test_label:]
CV_diff_test = CV_diff[:test_label]
CV_diff_train = CV_diff[test_label:]
C_times_V_test = C_times_V[:test_label]
C_times_V_train = C_times_V[test_label:]
real_gfjf_test = real_gfjf[:test_label]
real_gfjf_train = real_gfjf[test_label:]
C2_RJY2_train = C2_RJY2[test_label:]
C2_RJY2_test = C2_RJY2[:test_label]

from sklearn.linear_model import LinearRegression
slr = LinearRegression()
#y=[Q[i]+C1_deltaV[i]-C2_deltaV[i]-CV_diff[i] for i in range(len(Q))]
y = [Q_train[i]+C1_deltaV_train[i]-C2_deltaV_train[i]-CV_diff_train[i] for i in range(len(Q_train))]
y = [Q_train[i]+C1_deltaV_train[i]-C2_deltaV_train[i]-CV_diff_train[i] for i in range(len(Q_train))]
#y=[Q[i]+C1_deltaV[i]-C2_deltaV[i] for i in range(len(Q))]
y = np.array(y)
X0 = np.array(C_times_V_train)
#X0=dc
y = y.reshape(-1, 1)
X0 = X0.reshape(-1, 1)

#X= scaler.fit_transform(X0)
X = X0
#print(X)
#print(y)
slr.fit(X,y)
y_pred=slr.predict(X)
print(slr.coef_)
print(slr.intercept_)
print(slr.score(X,y))

k = 0.9478
b = 296.4306


#gfjf_predict=[(C2_deltaV_test[i]-C1_deltaV_test[i]+C_times_V_test[i]*k+CV_diff_test[i]+b) for i in range(len(Q_test))]
#gfjf_predict=[(C2_deltaV_test[i]-C1_deltaV_test[i]+C_times_V_test[i]*slr.coef_[0][0]+CV_diff_test[i]+slr.intercept_[0])/(1.43*1000*0.21*0.25*(2/44)) for i in range(len(Q_test))]

#gfjf_predict_test1=[(C2_deltaV_test[i]-C1_deltaV_test[i]+C2_RJY2_test[i]*777.6*slr.coef_[0][0]+CV_diff_test[i]+slr.intercept_[0]) for i in range(len(Q_test))]
#gfjf_predict_test2=[(C2_deltaV_test[i]-C1_deltaV_test[i]+C_times_V_test[i]*slr.coef_[0][0]+CV_diff_test[i]+slr.intercept_[0]) for i in range(len(Q_test))]
#gfjf_predict=[(C2_deltaV[i]-C1_deltaV[i]+X[i]*slr.coef_[0][0]+slr.intercept_[0])/(1.43*1000*0.2*0.25*(2/44)) for i in range(len(Q))]
#print(1-slr.coef_[0][0])
#print(1.43*1000*0.21*0.25*(2/44))
'''
for i in range(len(gfjf_predict_test1)):
    print(C2_RJY2_test[i]*777.6,'   ',C_times_V_test[i],'   ',Q_test[i],"   ",Q_test[i]-gfjf_predict_test2[i])
    print(real_gfjf_test[i],'   real')
    print(gfjf_predict_test1[i],"  test1")
    print(gfjf_predict_test2[i],"  test2  ")
    print(gfjf_predict[i],"   pred")
    if gfjf_predict_test1[i]>0:
        print(gfjf_predict_test1[i]/(1.43*1000*0.21*0.25*(2/44)*(1-slr.coef_[0][0])),'   pred2')
'''

#计算新的SV

c1_list = []
c2_list = []
gfjf_list = []
num = 0
import random
#计算新的SV
def sv_pre_cal(c1,c2,cv_diff,sv_real,c1dv,c2dv):
    #print('enter sv_pre_cal: ',c1,c2,cv_diff,sv_real)
    a1 = c2*dV-c1*dV+slr.intercept_[0]
    #print("formula a1:  ", c2dv-c1dv+slr.intercept_[0])
    #print("a1: ",a1)
    delta_cvtest = (c2dv/dV-c2)*777.6
    delta01 = (c2dv-c1dv+slr.intercept_[0])-a1+delta_cvtest

    sv_pre_test = sv_real-(delta01/0.8)/(1.43*1000*0.21*0.25*(2/44))
    #由原始方程求解误差过大，所以将原方程和调整RJY后的作对比进行估计
    #print(delta01,sv_pre_test)
    sv_pre = sv_pre_test
    return sv_pre


for i in range(len(C2_deltaV)):
    #1.8，0.4为两个阈值
    if C2_deltaV[i]/dV <= 1.5 and C2_deltaV[i]/dV >= 0.4:
        c1_list.append(RJY1[i])
        c2_list.append(RJY2[i])
        gfjf_list.append(real_gfjf[i])
    #削峰，降到1.8
    elif C2_deltaV[i]/dV > 1.5:
        #超过阈值，对应调整RJY1,RJY2,计算新的sv值
        #print("i = ",i,'   c1 = ',C1_deltaV[i]/dV,'   c2 = ',C2_deltaV[i]/dV)
        c2 = 1.5-0.1*random.random()
        #print(c2)
        c2_list.append(c2)
        c1 = 1.5+1.2+0.1*random.random()
        #print(c1)
        c1_list.append(c1)
        sv_pre=sv_pre_cal(c1, c2, CV_diff[i],real_gfjf[i],C1_deltaV[i],C2_deltaV[i])
        gfjf_list.append(sv_pre)
        print('1.5')
        print(c1, c2, sv_pre, real_gfjf[i])
        num+=1
    #填谷，0.4以下提升到0.9
    else:
        c2 = 0.9+0.1*random.random()
        #print(c2)
        c2_list.append(c2)
        c1 = 0.9+1.2+0.1*random.random()
        #print(c1)
        c1_list.append(c1)
        sv_pre = sv_pre_cal(c1,c2,CV_diff[i],real_gfjf[i],C1_deltaV[i],C2_deltaV[i])
        gfjf_list.append(sv_pre)
        print('0.9')
        print(c1, c2, sv_pre, real_gfjf[i])
        num += 1

#输出
result = pd.DataFrame(columns=['new_c1','new_c2','new_gfjf'])
result['created_at'] = QuarterData['created_at'][:-1]
result['new_c1'] = c1_list
result['new_c2'] = c2_list
result['new_gfjf'] = gfjf_list
result['RJY1'] = np.delete(RJY1, [-1])
result['RJY2'] = np.delete(RJY2, [-1])
result['GFJF'] = np.delete(GFJF, [-1])
order = ['created_at', 'GFJF', 'new_gfjf', 'RJY1', 'RJY2', 'new_c1', 'new_c2']
result = result[order]
print('result---\n', result.columns)
print('result---', result)
#result.to_csv('D:\\qingxi\\gfj\\k_regressor_1.5.csv', index=None)


'''
def gfjf_plot(gfjf_predict,real_gfjf):
    X1 = []
    y1 = []
    X2 = []
    y2 = []
    record = []
    for i in range(len(gfjf_predict)):
        #print(abs(gfjf_predict[i] - real_gfjf[i]) / real_gfjf[i])
        if abs(gfjf_predict[i] - real_gfjf[i]) / real_gfjf[i] <= 0.15:
            X1.append(real_gfjf[i])
            y1.append(gfjf_predict[i])

        else:
            X2.append(real_gfjf[i])
            y2.append(gfjf_predict[i])
    print(len(X1),len(gfjf_predict))
    print(len(X1)/len(gfjf_predict))
    _=error_calculate(gfjf_predict,real_gfjf)

    plt.scatter(X1, y1, c='red',label='acceptable')
    plt.scatter(X2, y2, c='blue',label='unacceptable')
    plt.xlabel('real')
    plt.ylabel('predict')
    plt.plot(real_gfjf, real_gfjf, c='purple', label='real=pred')
    #plt.plot(x,gfjf_predict,c='blue',label='pred')
    plt.legend()
    plt.show()
    x = np.linspace(1, len(real_gfjf), len(real_gfjf))
    plt.plot(x, real_gfjf, c='red', label='real')
    plt.plot(x, gfjf_predict, c='blue', label='pred')
    #plt.plot(x,gfjf_predict,c='blue',label='pred')
    plt.legend()
    plt.show()
#gfjf_plot(gfjf_predict,real_gfjf)
#gfjf_plot(gfjf_predict,real_gfjf_test)

plt.scatter(X , y , c = 'blue')
plt.plot(X,y_pred , c = 'red' , linewidth = 2)
plt.xlabel('real')
plt.ylabel('predict')
plt.legend()
plt.tight_layout()
plt.show()
'''
