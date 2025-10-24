import pandas as pd

import pandas as pd
import numpy as np
import pymysql

import tensorflow as tf
from keras.models import Sequential, load_model
import joblib
import time
import gc
import schedule
import datetime
import matplotlib.pyplot as plt
import pywt
import copy


def adjust_data1(data_4):
    """
    修改异常的数据
    :param data_4:
    :return:
    """

    feature = ['SQP2_SHC_ORP', 'SQP2_SHC_ORP_L2', 'PLC2_SHC_B_SS1', 'PLC2_SHC_B_SS2',
               'PLC2_WNBF_WHL_LL', 'PLC2_SHC_B_RJY1', 'PLC2_SHC_B_RJY2']
    data_dif = [1.8, 19, 43, 150, 100, 0.5, 0.4]
    data_3 = copy.deepcopy(data_4)
    # data_3.dropna()
    # data_3['PLC2_SHC_B_SS2_diff'] = data_3['PLC2_SHC_B_SS2'].diff()
    # data_3['PLC2_SHC_B_SS2_diff'] = data_3['PLC2_SHC_B_SS2_diff'].fillna(0)
    #
    # data_3['PLC2_WNBF_WHL_LL_diff'] = data_3['PLC2_WNBF_WHL_LL'].diff()
    # data_3['PLC2_WNBF_WHL_LL_diff'] = data_3['PLC2_WNBF_WHL_LL'].fillna(0)
    #
    # print('max--',data_3['PLC2_SHC_B_SS2_diff'].max())
    # print('min--', data_3['PLC2_SHC_B_SS2_diff'].min())
    # print('mean--', data_3['PLC2_SHC_B_SS2_diff'].mean())
    # print('mean--', abs(data_3['PLC2_SHC_B_SS2_diff']).mean())
    # print('describe---\n', data_3['PLC2_SHC_B_SS2_diff'].describe())
    # print('iloc-取---\n', data_3.iloc[1868:1870, 18])
    # print('均值-----\n', data_3.loc[1868:1870, 'PLC2_SHC_B_SS2'])
    # print('均值-----', data_3.loc[1868:1870, 'PLC2_SHC_B_SS2'].mean())
    # print('均值-减----\n', data_3.loc[1870-3:1870-1, 'PLC2_SHC_B_SS2'])
    # data_index = data_3[abs(data_3['PLC2_SHC_B_SS2_diff']) > 150].index
    # # data_index = data_3[abs(data_3['PLC2_SHC_B_SS2_diff']) > data_dif[3]].index
    # print('索引----\n', data_index)
    # print('索引----\n', type(data_index))
    #
    # data_index_1 = data_3[abs(data_3['PLC2_SHC_B_SS2_diff']) > data_dif[3]].index
    # print('---索引---1---\n', data_index_1)
    #
    # data_index_2 = data_3[abs(data_3['PLC2_WNBF_WHL_LL_diff']) > data_dif[3]].index
    # print('---索引--污泥-1---\n', data_index_2)
    #
    # print('改前---', data_3.loc[1870, 'PLC2_SHC_B_SS2'])
    # data_3.loc[1870, 'PLC2_SHC_B_SS2'] = 123
    # data_3.loc[1870, 'PLC2_SHC_B_SS2'] = data_3.loc[1870-3:1870-1, 'PLC2_SHC_B_SS2'].mean()
    # print('改后---', data_3.loc[1870,'PLC2_SHC_B_SS2'])
    # #按差分算
    # # data_3.loc[1870, 'PLC2_SHC_B_SS2'] = data_3.loc[1870, 'PLC2_SHC_B_SS2'] + data_3.loc[1870 - 4:1870 - 1, 'PLC2_SHC_B_SS2_diff'].mean()
    # # print('改后加上差分---', data_3.loc[1870, 'PLC2_SHC_B_SS2'])
    # print('改前---', data_3.loc[1871, 'PLC2_SHC_B_SS2'])
    # data_3.loc[1871, 'PLC2_SHC_B_SS2'] = 124
    # data_3.loc[1871, 'PLC2_SHC_B_SS2'] = data_3.loc[1871 - 3:1871 - 1, 'PLC2_SHC_B_SS2'].mean()
    # print('改后---', data_3.loc[1871, 'PLC2_SHC_B_SS2'])
    # # data_3.loc[1871, 'PLC2_SHC_B_SS2'] = data_3.loc[1871, 'PLC2_SHC_B_SS2'] + data_3.loc[1871 - 3:1871 - 1, 'PLC2_SHC_B_SS2_diff'].mean()
    # # print('改后加上差分---', data_3.loc[1871, 'PLC2_SHC_B_SS2'])
    #
    #
    #
    #
    # for i in range(len(data_index)):
    #     print('index====', data_index[i])
    #
    #     if data_index[i] - 3 < 0:
    #         data_3.loc[data_index[i], 'PLC2_SHC_B_SS2'] = data_3.loc[0:data_index[i] - 1, 'PLC2_SHC_B_SS2'].mean()
    #     else:
    #         print('改前---', data_3.loc[data_index[i], 'PLC2_SHC_B_SS2'])
    #         data_3.loc[data_index[i], 'PLC2_SHC_B_SS2'] = data_3.loc[data_index[i] - 3:data_index[i] - 1,'PLC2_SHC_B_SS2'].mean()
    #         print('改后---', data_3.loc[data_index[i], 'PLC2_SHC_B_SS2'])
    #
    # plt.plot(data_4['PLC2_SHC_B_SS2'], color='r', label='true')
    # plt.plot(data_3['PLC2_SHC_B_SS2'], color='b', label='new')
    # # plt.xlim((0, 8000))
    # # plt.ylim((0, 500))     设置  x
    # plt.legend()
    # plt.title('adjust_data')
    # plt.xlabel("data")
    # plt.ylabel('predict-1')
    # plt.show()
    # #
    for i in range(len(feature)):
        new_fea = feature[i] + '_diff'
        print('----feature----', new_fea)
        data_3[new_fea] = data_3[feature[i]].diff()
        data_3[new_fea] = data_3[new_fea].fillna(0)
        print('max--', data_3[new_fea].max())
        print('min--', data_3[new_fea].min())
        print('mean--', data_3[new_fea].mean())
        print('abs_mean--', abs(data_3[new_fea]).mean())
        data_index = data_3[abs(data_3[new_fea]) > data_dif[i]].index

        for j in range(len(data_index)):
            print('index====', data_index[j])
            if data_index[j] - 3 < 0:
                data_3.loc[data_index[j], feature[i]] = data_3.loc[0:data_index[j] - 1, feature[i]].mean()
            else:
                print('改前---', data_3.loc[data_index[j], feature[i]])
                data_3.loc[data_index[j], feature[i]] = data_3.loc[data_index[j] - 3:data_index[j] - 1, feature[i]].mean()
                print('改后---', data_3.loc[data_index[j], feature[i]])

        # plt.plot(data_4[feature[i]], color='r', label='true')
        # plt.plot(data_3[feature[i]], color='b', label='new')
        # # plt.xlim((0, 8000))
        # # plt.ylim((0, 500))     设置  x
        # plt.legend()
        # plt.title(feature[i])
        # plt.xlabel("data")
        # plt.ylabel('predict-1')
        # plt.show()

    return data_3


def plot_signal_decomp(data, w, title):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    其中D1,D2,…,DN分别为第一层、第二层到等N层分解得到的高频信号，AN为第N层分解得到的低频信号。

    """
    mode = pywt.Modes.smooth
    w = pywt.Wavelet(w)  # 选取小波函数
    a = data
    N=len(a)
    ca = []  # 近似分量
    cd = []  # 细节分量
    for i in range(5):
        (a, d) = pywt.dwt(a, w, mode)  # 进行5阶离散小波变换
        ca.append(a)
        cd.append(d)
    print('---ca近似分量长度--', len(ca[0]))
    print('---cd细节分量长度---', len(cd))
    print('type--ca---', type(ca))

    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        # rec_a.append(pywt.waverec(coeff_list, w))  # 重构
        rec_a.append(pywt.waverec(coeff_list, w)[0:N])  # 重构
    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        if i == 3:
            print('len_coeff----',len(coeff))
            print('len_coeff_list----',len(coeff_list))
        #rec_d.append(pywt.waverec(coeff_list, w))
        rec_d.append(pywt.waverec(coeff_list, w)[0:N])

    print('重构-细节-rec_a--\n',len(rec_a))
    print('重构-高频-rec_d--\n', len(rec_d))
    print('重构-高频-rec_a--\n', len(rec_a[0]))
    print('重构-高频-rec_d--\n', len(rec_d[0]))

    # fig = plt.figure()
    # ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    # ax_main.set_title(title)
    # ax_main.plot(data)
    # ax_main.set_xlim(0, len(data) - 1)
    #
    # for i, y in enumerate(rec_a):
    #     #print('细节---\n',i)
    #     ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
    #     ax.plot(y, 'r')
    #     ax.set_xlim(0, len(y) - 1)
    #     ax.set_ylabel("A%d" % (i + 1))
    #
    # for i, y in enumerate(rec_d):
    #     ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
    #     ax.plot(y, 'g')
    #     ax.set_xlim(0, len(y) - 1)
    #     ax.set_ylabel("D%d" % (i + 1))

    return rec_a

# def create_data1(dataset, gfj_back):
#     """
#         8点的样本里 需要 8点半的风量，错开一个时间点
#     :param dataset:
#     :param gfj_back:  错开的间隔
#     :return:
#     """
#     new_data_1 = []
#     for i in range(len(dataset)-gfj_back):
#         dx = dataset[i, 0:9]
#         # print('data---gfj--shape---', dx.shape)
#         #print('type-gfj----', type(dx))
#         dx = dx.reshape(-1, 9)
#         # print('data---gfj--shape---', dx.shape)
#         data_gfj = dataset[i+gfj_back, -3]
#         # print('data---gfj--shape---', data_gfj)
#         # print('type-----', type(data_gfj))
#         # print('data---gfj--shape---', data_gfj.shape)
#         data_gfj = data_gfj.reshape(-1, 1)
#         # print('data---gfj--shape---', data_gfj.shape)
#         # print('data---gfj--shape---', type(data_gfj))
#         data_do = dataset[i, -2:]
#         data_do = data_do.reshape(-1, 2)
#         #print('')
#         #data_c = np.concatenate([dx, data_gfj, data_do], axis=1)
#         data_c = np.hstack((dx, data_gfj, data_do))
#         # print('type--data-c--',type(data_c))
#         # print('datac--shape--',data_c.shape)
#         new_data_1.append(data_c)
#     return np.array(new_data_1)


def create_data1(dataset, gfj_back):
    """
        8点的样本里 需要 8点半的风量，错开一个时间点
    :param dataset:
    :param gfj_back:  错开的间隔
    :return:
    """
    new_data_1 = []
    for i in range(gfj_back, len(dataset)):
        dx = dataset[i-gfj_back, 0:9]
        # print('data---gfj--shape---', dx.shape)
        #print('type-gfj----', type(dx))
        dx = dx.reshape(-1, 9)
        # print('data---gfj--shape---', dx.shape)
        data_gfj = dataset[i, -3]
        # print('data---gfj--shape---', data_gfj)
        # print('type-----', type(data_gfj))
        # print('data---gfj--shape---', data_gfj.shape)
        data_gfj = data_gfj.reshape(-1, 1)
        # print('data---gfj--shape---', data_gfj.shape)
        # print('data---gfj--shape---', type(data_gfj))
        data_do = dataset[i-gfj_back, -2:]
        data_do = data_do.reshape(-1, 2)
        #print('')
        #data_c = np.concatenate([dx, data_gfj, data_do], axis=1)
        data_c = np.hstack((dx, data_gfj, data_do))
        # print('type--data-c--',type(data_c))
        # print('datac--shape--',data_c.shape)
        new_data_1.append(data_c)
    return np.array(new_data_1)

# LSTM 的数据格式
# def create_dataset(dataset, look_back, pre_step):
#     """
#     构建LSTM 的输入样本shape
#     :param dataset:
#     :param look_back:
#     :param pre_step:  预测的步长
#     :return:
#     """
#
#     dataX, dataY = [], []
#     for i in range(len(dataset) - look_back - pre_step):
#         a = dataset[i:(i + look_back), :]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back + pre_step-1, -1])
#         #dataY.append(dataset[i + look_back-1, -1])
#     return np.array(dataX), np.array(dataY)


def create_dataset(dataset, look_back):
    """
    构建LSTM 的输入样本shape
    :param dataset:
    :param look_back:
    :return:
    """

    dataX, dataY = [], []
    for i in range(len(dataset) - look_back + 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
    return np.array(dataX)


def rjy_sql(sql_ip, predict, dt, sv_old, sv_new, sv_di):
    print('插入数据')
    conn = pymysql.connect(host=sql_ip,
                           user='lwq',
                           passwd='e6544d2cd5f03f2a502Daq54@6!aDq0k9f8001',
                           port=3306,
                           db='lwq',
                           charset='utf8')
    cur = conn.cursor()  # 创建游标对象
    #insert_emp_sql = "insert into pre_15min_rjy(pre_rjy2, pre_date, old_sv, new_sv) values(%s, %s, %s, %s)"
    insert_emp_sql = "insert into pre_15min_rjy4(pre_rjy2, pre_date, old_sv, new_sv, sv_dif) values(%s, %s, %s, %s,%s)"

    cur.execute(insert_emp_sql, [predict, dt, sv_old, sv_new, sv_di])
    # 提交数据
    conn.commit()
    #关闭 cursor对象
    cur.close()
    #关闭 connection对象
    conn.close()


def adjust_sv(rjy):
    """
    根据预测 rjy 调整 sv
    :param rjy:
    :return:
    """

    conn_gfj1 = pymysql.connect(host='192.168.18.23',
                                user='readOnly',
                                # user='gl',
                                passwd='P@sswor@dro123',
                                port=3306,
                                db='sq',
                                charset='utf8')

    #sql = 'select created_at, `SQ-T2-GFJ1-40026`, `SQ-T2-GFJ1-40100`,`SQ-T2-GFJ2-40026`, `SQ-T2-GFJ2-40100`,`SQ-T2-GFJ3-40026`, `SQ-T2-GFJ3-40100`,`SQ-T2-GFJ4-40026`, `SQ-T2-GFJ4-40100` from sq_gfj4_record_v2_202305 order by created_at desc limit 1'
    sql = 'select created_at, `GFJ1_ZT`, `GFJ1_SVSDZ`,`GFJ2_ZT`, `GFJ2_SVSDZ`, `GFJ3_ZT`, `GFJ3_SVSDZ`, `GFJ4_ZT`, `GFJ4_SVSDZ` from sq_gfj_record_202305 order by created_at desc limit 1'
    gfj2_sv = pd.read_sql(sql, conn_gfj1)
    # time_gfj2 = gfj2_sv['SQ-T2-GFJ4-40100']
    # run_1 = gfj2_sv['SQ-T2-GFJ4-40026']
    time_gfj2 = gfj2_sv['GFJ1_SVSDZ'].values
    run_1 = gfj2_sv['GFJ1_ZT']
    time_gfj2_1 = time_gfj2
    print('40100---', time_gfj2)
    print('40026---', run_1)
    print('40100---', type(time_gfj2))
    print('40026---', type(run_1))

    sum_gd = 0
    if float(gfj2_sv['GFJ1_ZT']) == 1:
        sum_gd += float(gfj2_sv['GFJ1_SVSDZ'])
    # if float(gfj2_sv['GFJ2_ZT']) == 1:
    #     sum_gd += float(gfj2_sv['GFJ2_SVSDZ'])
    # if float(gfj2_sv['GFJ3_ZT']) == 1:
    #     sum_gd += float(gfj2_sv['GFJ3_SVSDZ'])
    # if float(gfj2_sv['GFJ4_ZT']) == 1:
    #     sum_gd += float(gfj2_sv['GFJ4_SVSDZ'])

    print('sum-gd--', sum_gd)
    # if data_gfj.loc[i, 'GFJ2_ZT'] == 1:
    #     sum_gd += data_gfj.loc[i, 'GFJ2_SVSDZ']
    #     sum_gl += data_gfj.loc[i, 'GFJ2_XHGL']
    # if data_gfj.loc[i, 'GFJ3_ZT'] == 1:
    #     sum_gd += data_gfj.loc[i, 'GFJ3_SVSDZ']
    #     sum_gl += data_gfj.loc[i, 'GFJ3_XHGL']
    # if data_gfj.loc[i, 'GFJ4_ZT'] == 1:
    #     sum_gd += data_gfj.loc[i, 'GFJ4_SVSDZ']
    #     sum_gl += data_gfj.loc[i, 'GFJ4_XHGL']
    # data_gfj.loc[i, 'sum_gl'] = sum_gl
    # data_gfj.loc[i, 'sum_gd'] = sum_gd

    time_gfj2_2 = 0
    sv_dif = 0   # 调整sv
    if rjy >= 3:
        print('---sv 降 10---')
        sv_dif = -10
        time_gfj2_2 = time_gfj2_1 - 100
        if time_gfj2_2 <= 450:
            time_gfj2_2 = 450

    elif 1.8 <= rjy < 3:
        print('---sv 降 5---')
        sv_dif = -5
        time_gfj2_2 = time_gfj2_1 - 50
        if time_gfj2_2 <= 450:
            time_gfj2_2 = 450

    elif 1.5 <= rjy < 1.8:
        print('---sv 降 3---')
        sv_dif = -3
        time_gfj2_2 = time_gfj2_1 - 30
        if time_gfj2_2 <= 450:
            time_gfj2_2 = 450

    elif 0.9 <= rjy < 1.5:
        print('---sv 不变---')
        sv_dif = 0
        time_gfj2_2 = time_gfj2_1
    elif 0.7 <= rjy < 0.9:
        print('---sv 升 3---')
        sv_dif = 3
        time_gfj2_2 = time_gfj2_1 + 30
        if time_gfj2_2 >= 900:
            time_gfj2_2 = 900
    elif 0.5 <= rjy < 0.7:
        sv_dif = 5
        print('---sv 升 5---')
        time_gfj2_2 = time_gfj2_1 + 50
        if time_gfj2_2 >= 900:
            time_gfj2_2 = 900

    elif rjy < 0.5:
        print('---sv 升 10---')
        sv_dif = 10
        time_gfj2_2 = time_gfj2_1 + 100
        if time_gfj2_2 >= 900:
            time_gfj2_2 = 900

    return time_gfj2_1, time_gfj2_2, sv_dif


def pre_do(test_new):
    """
             不改变风量 预测do
    :param test_new:  计算的数据
    :return:
    """

    model = load_model('D:\\Users\\Administrator\\PycharmProject\\shuangqing_DO\\model_rjy2\\rjy2_wavelet_15min_step4_12.h5')
    print('---model.summary--')
    model.summary()

    predict_do = model.predict(test_new)
    return predict_do


def flow_and_sv():
    """
    读取本地数据库数据进行计算，并把预测rjy存到
    :return:
    """
    conn = pymysql.connect(host='192.168.18.23',
                           user='readOnly',
                           # user='gl',
                           passwd='P@sswor@dro123',
                           port=3306,
                           db='sq',
                           charset='utf8')
    cur = conn.cursor()  # 创建游标对象
    # sql = "SELECT created_at,AI_P1_INCOME_SH_TN, PLC1_CCC_LL_FLOW,`SQP2-SHC-WaterTemp` as SQP2_SHC_WaterTemp, `SQP2-SHC-ORP` as SQP2_SHC_ORP, " \
    #       "`SQP2-SHC-ORP-L2` as SQP2_SHC_ORP_L2, PLC2_SHC_B_RJY1,PLC2_SHC_B_SS1, PLC2_SHC_B_SS2, PLC2_WNBF_WHL_LL, PLC3_YSN_LL_FLOW2, " \
    #       "PLC2_GFJF_LL_FLOW, PLC2_SHC_B_RJY2 FROM sq_gfj_record_202210 where created_at BETWEEN '2022-10-02 22:28:20' AND '2022-10-03 02:00:00'"

    sql = "SELECT created_at,AI_P1_INCOME_SH_TN, PLC1_CCC_LL_FLOW,`SQP2-SHC-WaterTemp` as SQP2_SHC_WaterTemp, `SQP2-SHC-ORP` as SQP2_SHC_ORP, " \
          "`SQP2-SHC-ORP-L2` as SQP2_SHC_ORP_L2, PLC2_SHC_B_RJY1,PLC2_SHC_B_SS1, PLC2_SHC_B_SS2, PLC2_WNBF_WHL_LL, PLC3_YSN_LL_FLOW2, " \
          "PLC2_GFJF_LL_FLOW, PLC2_SHC_B_RJY2 FROM sq_gfj_record_202211 order by created_at desc limit 195"


    print("---开始查询数据库---")

    feature_s = ['AI_P1_INCOME_SH_TN', 'PLC1_CCC_LL_FLOW', 'SQP2_SHC_WaterTemp', 'SQP2_SHC_ORP', 'SQP2_SHC_ORP_L2',
                 'PLC2_SHC_B_SS1', 'PLC2_SHC_B_SS2', 'PLC2_WNBF_WHL_LL', 'PLC3_YSN_LL_FLOW2', 'PLC2_GFJF_LL_FLOW', 'PLC2_SHC_B_RJY1', 'PLC2_SHC_B_RJY2']

    print('len---', len(feature_s))
    a = pd.read_sql(sql, conn)

    print("a ====\n ", a)
    a = a.dropna()
    print('aaaa---', a)
    for i in range(len(feature_s)):
        print('字段---', feature_s[i])
        # a[feature3[i]] = a[feature3[i]].apply(lambda x: float(x))
        a[feature_s[i]] = a[feature_s[i]].astype('float')
    print('a----\n', a)

    a = adjust_data1(a)
    print('len---data_new', len(a))

    new_data = copy.deepcopy(a)
    print('数据长度--len\n', len(new_data))
    print('len---new_data===', len(new_data))
    for j in feature_s:
        print('字段====', j)
        rec_aa = plot_signal_decomp(a.loc[:, j], 'db4', j)  # sym5 sym7 db4 db5
        rec = rec_aa[3]
        print('len--rec_aa--', len(rec_aa[3]))
        print('rec----\n', type(rec_aa[3]))
        # new_data.loc[:, i] = rec[:-1]
        new_data.loc[:, j] = rec
    print('小波变换后的数据长度--', len(new_data))
    print('小波变换后的数据--\n', new_data)

    feature_s1 = ['created_at', 'AI_P1_INCOME_SH_TN', 'PLC1_CCC_LL_FLOW', 'SQP2_SHC_WaterTemp', 'SQP2_SHC_ORP', 'SQP2_SHC_ORP_L2',
                 'PLC2_SHC_B_SS1', 'PLC2_SHC_B_SS2', 'PLC2_WNBF_WHL_LL', 'PLC3_YSN_LL_FLOW2', 'PLC2_GFJF_LL_FLOW', 'PLC2_SHC_B_RJY1', 'PLC2_SHC_B_RJY2']
    #a = new_data
    a = new_data.loc[:, feature_s1]
    print('type--aaa--', type(a))
    a['created_at'] = pd.to_datetime(a['created_at'])
    a = a.set_index('created_at')
    a = a.resample('15min').mean()
    df1 = a
    print('---df1---', df1)
    print('len=====df1===\n', len(df1))

    # dt1 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dt1 = (datetime.datetime.now() + datetime.timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M:%S")

    df1 = np.array(df1)
    new_data1 = create_data1(df1, 1)
    print('len_new_data1++++++\n', len(new_data1))
    print('---new_data1----\n',new_data1)
    new_data1 = new_data1.reshape(-1, 12)

    st_model = joblib.load('D:\\Users\\Administrator\\PycharmProject\\shuangqing_DO\\model\\stand_wavelet_15min_step4_12')

    dataset1 = st_model.transform(new_data1[:, :-1])           # 只对特征归一化
    dataset1 = np.concatenate((dataset1, new_data1[:, -1].reshape(-1, 1)), axis=1)
    test_new = create_dataset(dataset1, 12)
    print('len====testnew===\n', len(test_new))
    # print('len====testnew===\n', test_new[-1].index)
    print('test_new-----\n', test_new)
    test_new1 = test_new[-1].reshape(-1, 12, 12)
    print('最后一个样本---\n', test_new1)
    print('最后一个样本---\n', test_new1.shape)
    no_flow_rjy = pre_do(test_new1)
    no_flow_rjy2 = float(no_flow_rjy)
    print('---加60min后的预测时间::', dt1)
    print('---预测的rjy:', no_flow_rjy2)
    #gfj_flow, pre_rjy = gfj_flow_compute(test_new1)
    #pre_rjy2 = float(pre_rjy)
    #gfj_flow2 = float(gfj_flow)
    # print("rong jie yang = ", pre_rjy2)

    #sv1, sv2, sv_dif = adjust_sv(no_flow_rjy2)
    print('准备插入数据')
    #rjy_sql('192.168.18.23', no_flow_rjy2, dt1, sv1, sv2, sv_dif)   #pre_15min_rjy4  #11.07
    #rjy_sql('192.168.18.23', no_flow_rjy2, dt1)
    gc.collect()


if __name__ == '__main__':
    flow_and_sv()
    schedule.every(15).minutes.do(flow_and_sv)
    #schedule.every(1).minutes.do(flow_and_sv)
    # while True:
    #     schedule.run_pending()
    #     time.sleep(2)



