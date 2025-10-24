
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#  合并 每个月的 sv-风数据
d_flow_sv_11 = pd.read_csv('D:\\qingxi\\gfj\\gfj_11_sv_flow.csv')
d_flow_sv_12 = pd.read_csv('D:\\qingxi\\gfj\\gfj_12_sv_flow.csv')
d_flow_sv_01 = pd.read_csv('D:\\qingxi\\gfj\\gfj_01_sv_flow.csv')
df_11 = pd.read_csv('D:\\qingxi\\data\\2022_11_01-30.csv')

df_new_do_gfj = pd.read_csv('D:\\qingxi\\gfj\\k_regressor_1.5.csv')
#df_new_do_gfj = pd.read_csv('D:\\qingxi\\gfj\\k_regressor_1.5.csv')
print('new---gfj', df_new_do_gfj.columns)
print('new---index', df_new_do_gfj.index)
print('new---len', len(df_new_do_gfj))

print('len--d_flow--', len(d_flow_sv_01))
df_sum = pd.concat([d_flow_sv_11, d_flow_sv_12, d_flow_sv_01], axis=0)
print('合并后的----sv--', len(df_sum))
df_sum = df_sum.dropna()
#df_sum.to_csv('D:\\qingxi\\gfj\\gfj_11-01_sv_flow.csv', index=None)
df_flow_range = pd.DataFrame(data=None, columns={'sv', 'flow_l', 'flow_h'})
print('df-flow==\n', df_flow_range)


for i in range(450, 1500, 10):
    print('iii---', i)
    data_s_f = df_sum[df_sum['sum_gd'] == i]
    print('data_s-f---\n', data_s_f.loc[:, ['sum_gd', 'PLC2_GFJF_LL_FLOW']])

    print('len--select---\n', len(data_s_f))
    if len(data_s_f) == 0:
        continue

    q1 = np.percentile(data_s_f['PLC2_GFJF_LL_FLOW'], 0.03)
    q3 = np.percentile(data_s_f['PLC2_GFJF_LL_FLOW'], 99)
    interval = q3 - q1
    #high = q3 + 1 * interval
    #low = q1 - 1.5 * interval
    high = q3
    low = q1
    # high = q3
    # low = q1
    print('sv=%d的上四分位数：%.2f  下四分位数：%.2f  间隔 %.2f ' % (i, q3, q1, interval))
    print('sv=%d范围为（%.2f,%.2f）' % (i, low, high))
    df_flow_range = df_flow_range.append({'sv': i, 'flow_l': low, 'flow_h': high}, ignore_index=True)
    # if i > 520:
    #     break
#df_flow_range.to_csv('D:\\qingxi\\gfj\\df_flow_range_11-01_sv_flow.csv', index=None)
print('df--flow--\n', df_flow_range.columns)
print('df__flow--\n', type(df_flow_range.loc[0, 'flow_l']))
print('-----type---', type(df_flow_range))
print('---type---', type(df_11))

# for i in range(len(df_11)):
#     # print('type---', type(df_12.loc[i, 'PLC2_GFJF_LL_FLOW']))
#     # print('对应的风----', df_12.loc[i, 'PLC2_GFJF_LL_FLOW'])
#     print('i---', i)
#     if i > 5:
#         break
#     for j in range(len(df_flow_range)):
#         if df_11.loc[i, 'PLC2_GFJF_LL_FLOW'] > df_flow_range.loc[j, 'flow_l'] and df_11.loc[i, 'PLC2_GFJF_LL_FLOW'] < df_flow_range.loc[j, 'flow_h']:
#             print('风---\n', df_11.loc[i, 'PLC2_GFJF_LL_FLOW'])
#             print('风---\n', df_flow_range.loc[j, 'sv'])
#             df_11.loc[i, 'new_sv'] = df_flow_range.loc[j, 'sv']
#             break
# print('new_df_12---', df_11.columns)


def electric_gd(data_ele):
    """
    按 谷、峰、平 求   电费
    :param data_ele:
    :return:
    """
    # data_ele = data_ele.dropna()
    #
    #data_ele['save_date'] = pd.to_datetime(data_ele['save_date'])
    data_ele['created_at'] = pd.to_datetime(data_ele['created_at'])
    #data_ele = data_ele.set_index('created_at')
    data_ele = data_ele.set_index('created_at')
    data_ele = data_ele.fillna(0)
    #data_ele = data_ele.set_index('created_at')
    #data_ele = data_ele.resample('3min').first()

    data_low = data_ele.between_time('0:00', '6:59:59')
    data_low = data_low.reset_index()

    data_low1 = data_ele.between_time('23:00', '23:59:59')
    data_low1 = data_low1.reset_index()

    low_concat = pd.concat([data_low, data_low1], axis=0)
    print('low---describe---\n', low_concat['sum_gd'].describe())


    data_mil = data_ele.between_time('7:00', '8:59:59')
    data_mil = data_mil.reset_index()

    data_mil1 = data_ele.between_time('12:00', '15:59:59')
    data_mil1 = data_mil1.reset_index()

    data_mil2 = data_ele.between_time('21:00', '22:59:59')
    data_mil2 = data_mil2.reset_index()

    mil_concat = pd.concat([data_mil, data_mil1, data_mil2], axis=0)
    print('mil---describe---\n', mil_concat['sum_gd'].describe())


    data_high = data_ele.between_time('9:00', '11:59:59')
    data_high = data_high.reset_index()

    data_high1 = data_ele.between_time('16:00', '20:59:59')
    data_high1 = data_high1.reset_index()
    high_concat = pd.concat([data_high, data_high1], axis=0)
    print('high---describe---\n', high_concat['sum_gd'].describe())
    low_concat = low_concat.sort_values(by='created_at')
    mil_concat = mil_concat.sort_values(by='created_at')
    high_concat = high_concat.sort_values(by='created_at')

    return low_concat, mil_concat, high_concat


def new_do_sv(data_gfj_old, data_gfj_new):
    """
    调整gfj前、后的， sv 对比
    :param data_gfj_old:
    :param data_gfj_new: 改后的新风量，已经下采样15min后的数据
    :return:
    """

    data_gfj_old['created_at'] = pd.to_datetime(data_gfj_old['created_at'])
    data_gfj_old = data_gfj_old.set_index('created_at')
    data_gfj_old = data_gfj_old.resample('15min').mean()
    data_gfj_old = data_gfj_old.reset_index()
    print('data---gfj--old--\n', data_gfj_old.loc[0:100, 'PLC2_GFJF_LL_FLOW'])
    print('data---gfj--new--\n', data_gfj_new.loc[0:100, 'GFJF'])
    df_new_gfj_sv = pd.concat([data_gfj_new, data_gfj_old.loc[:, 'sum_gd']], axis=1)

    df_new_gfj_sv['gfj_dif'] = df_new_gfj_sv['GFJF'] - df_new_gfj_sv['new_gfjf']
    df_new_gfj_sv['do_dif'] = df_new_gfj_sv['RJY2'] - df_new_gfj_sv['new_c2']
    print('df_new---gfj_sv--\n', df_new_gfj_sv.columns)

    df_sv_p = df_new_gfj_sv[df_new_gfj_sv['do_dif']!=0]
    #df_sv_p = df_new_gfj_sv[df_new_gfj_sv['gfj_dif'] > 0]
    #df_sv_p = df_sv_p[df_sv_p['new_gfjf'] > 0]
    df_sv_p = df_sv_p.reset_index(drop=True)
    #df_sv_p.to_csv('D:\\qingxi\\gfj\\do_1.5_dif.csv', index=None)

    for i in range(len(df_sv_p)):
        # print('type---', type(df_12.loc[i, 'PLC2_GFJF_LL_FLOW']))
        # print('对应的风----', df_12.loc[i, 'PLC2_GFJF_LL_FLOW'])
        print('i---', i)
        for j in range(len(df_flow_range)):
            if df_sv_p.loc[i, 'new_gfjf'] > df_flow_range.loc[j, 'flow_l'] and df_sv_p.loc[i, 'new_gfjf'] < df_flow_range.loc[j, 'flow_h']:
                print('风---\n', df_sv_p.loc[i, 'new_gfjf'])
                print('old_sv---\n', df_sv_p.loc[i, 'sum_gd'])
                print('new_sv---\n', df_flow_range.loc[j, 'sv'])
                df_sv_p.loc[i, 'new_sv'] = df_flow_range.loc[j, 'sv']
                break

    df_sv_p['sv_dif'] = df_sv_p['sum_gd'] - df_sv_p['new_sv']
    #df_sv_p = df_sv_p[df_sv_p['sv_dif'] > 0]
    t = 0.25
    gfj_p = df_sv_p['sv_dif'].sum() / 10
    print('len时间---', t)
    print('sv_dif--sum-\n', gfj_p)
    print('节省电量--\n', t*gfj_p)
    print('节省电费--\n', t * gfj_p * 0.75)
    return df_sv_p


def plot_new_sv_do(new_data):
    """
    画 方程生成的新do、 sv 和 风量
    :param new_data:
    :return:
    """
    #new_data = new_data[new_data['new_gfjf'] > 0]
    #new_data = new_data[new_data['gfj_dif'] > 0]
    #new_data = new_data[new_data['sv_dif'] > 0]
    #new_data = new_data.reset_index(drop=True)

    plt.plot(new_data.loc[:, 'sum_gd'], color='r', label='gd_sv')
    plt.plot(new_data.loc[:, 'new_sv'], color='b', label='new_sv')
    x = np.arange(0, len(new_data))
    plt.fill_between(x, new_data.loc[:, 'sum_gd'], new_data.loc[:, 'new_sv'], where=(new_data.loc[:, 'new_sv'] < new_data.loc[:, 'sum_gd']), color='yellow')
    plt.fill_between(x, new_data.loc[:, 'sum_gd'], new_data.loc[:, 'new_sv'], where=(new_data.loc[:, 'new_sv'] > new_data.loc[:, 'sum_gd']), color='hotpink')

    # plt.xlim((0, 8000))
    # plt.ylim((0, 500))     设置  x
    plt.legend()
    plt.xlabel("data")
    plt.ylabel('sv')
    plt.show()

    plt.plot(new_data.loc[:, 'GFJF'], color='c', label='gfj_flow')
    plt.plot(new_data.loc[:, 'new_gfjf'], color='brown', label='new_flow')
    x = np.arange(0, len(new_data))
    plt.fill_between(x, new_data.loc[:, 'GFJF'], new_data.loc[:, 'new_gfjf'], where=(new_data.loc[:, 'new_gfjf'] < new_data.loc[:, 'GFJF']), color='yellow')
    plt.fill_between(x, new_data.loc[:, 'GFJF'], new_data.loc[:, 'new_gfjf'], where=(new_data.loc[:, 'new_gfjf'] > new_data.loc[:, 'GFJF']), color='hotpink')


    # plt.xlim((0, 8000))
    # plt.ylim((0, 500))     设置  x
    plt.legend()
    plt.xlabel("data")
    plt.ylabel('flow')
    plt.show()

    plt.plot(new_data.loc[:, 'RJY2'], color='green', label='do2')
    plt.plot(new_data.loc[:, 'new_c2'], color='aqua', label='new_do2')
    x = np.arange(0, len(new_data))
    plt.fill_between(x, new_data.loc[:, 'RJY2'], new_data.loc[:, 'new_c2'], where=(new_data.loc[:, 'new_c2'] < new_data.loc[:, 'RJY2']), color='yellow')
    plt.fill_between(x, new_data.loc[:, 'RJY2'], new_data.loc[:, 'new_c2'], where=(new_data.loc[:, 'new_c2'] > new_data.loc[:, 'RJY2']), color='hotpink')
    # plt.xlim((0, 8000))
    plt.ylim((0, 6))     #设置  x
    plt.legend()
    plt.xlabel("data")
    plt.ylabel('do')
    plt.show()


print('df---flow===\n', df_flow_range)
df11 = new_do_sv(d_flow_sv_11, df_new_do_gfj)
print('df-11.colunms--', df11.columns)
print('df——11', df11)
print('df——11', len(df11))
#df11.to_csv('D:\\qingxi\\gfj\\new_model_dv_dif.csv', index=None)

plot_new_sv_do(df11)

low_1, mil_1, high_1 = electric_gd(df11)
print('low-1---\n', len(low_1))
print('mil-1---\n', len(mil_1))
print('high_1---\n', len(high_1))
print('high---1--\n')
print('---len--df11---', len(df11))
print('---len--d---', len(df11)*0.25/24)
print('--low--', len(low_1))
low_e = low_1['sv_dif'].sum() / 10
low_c = low_e*0.25*0.38
print('low--省电费--', low_c)
print('low--省电量--', low_e*0.25)
print('--mil--', len(mil_1))
mil_e = mil_1['sv_dif'].sum() / 10
mil_c = mil_e*0.25*0.75
print('mil--省电费--', mil_c)
print('mil--省电量--', mil_e*0.25)
print('--high--', len(high_1))
high_e = high_1['sv_dif'].sum() / 10
print('highe-dif--', high_e)
high_c = high_e*0.25*1.09
print('high--省电量--', high_e*0.25)
print('high--省电费--', high_c)
print('省总电量---', low_e*0.25+mil_e*0.25+high_e*0.25)
print('省总电费---', low_c+mil_c+high_c)


low_e = low_1['sum_gd'].sum() / 10
low_c_1 = low_e*0.25*0.38
print('low-old-电费--', low_c_1)
print('low-old-电量--', low_e*0.25)

print('--mil--', len(mil_1))
mil_e = mil_1['sum_gd'].sum() / 10
mil_c_1 = mil_e*0.25*0.75
print('mil-old-电费--', mil_c_1)
print('mil-电量--', mil_e*0.25)

print('--high--', len(high_1))
high_e = high_1['sum_gd'].sum() / 10
print('high-gd-', high_e)
print('high-电量-', high_e*0.25)
high_c_1 = high_e*0.25*1.09
print('high-old-电费--', high_c_1)

old_c = low_e*0.25 + mil_e*0.25 + high_e*0.25


low_e = low_1['new_sv'].sum() / 10
low_c_2 = low_e*0.25*0.38
print('low-new-电费--', low_c_2)
print('low-new-电量--', low_e*0.25)
print('--mil--', len(mil_1))

mil_e = mil_1['new_sv'].sum() / 10
mil_c_2 = mil_e*0.25*0.75
print('mil-new-电费--', mil_c_2)
print('mil-new-电量--', mil_e*0.25)

print('--high--', len(high_1))
high_e = high_1['new_sv'].sum() / 10
print('high-new-', high_e)
high_c_2 = high_e*0.25*1.09
print('high-new-电量', high_e*0.25)
print('high-new-电费--', high_c_2)

new_c = low_e*0.25 + mil_e*0.25 + high_e*0.25

print('调整前总电费--', high_c_1+mil_c_1+low_c_1)
print('调整后总电费--', high_c_2+mil_c_2+low_c_2)
print('省出-总电费--', high_c_1+mil_c_1+low_c_1-high_c_2-mil_c_2-low_c_2)
print('省出-总电费占比--', (high_c_1+mil_c_1+low_c_1-high_c_2-mil_c_2-low_c_2)/(high_c_1+mil_c_1+low_c_1))

print('调整前总电量为old_c--', old_c)
print('调整后总电量为new_c--', new_c)
print('省出-总电量--', old_c - new_c)
print('省出-总电量占比--', (old_c - new_c)/old_c)

print('谷时段省电占比', low_1['sv_dif'].sum()/low_1['sum_gd'].sum())
print('平时段省电占比', mil_1['sv_dif'].sum()/mil_1['sum_gd'].sum())
print('峰时段省电占比', high_1['sv_dif'].sum()/high_1['sum_gd'].sum())


#a = (2.5+19.5+31.8+33.7+30.8+12.1+34.8+22.9+14.3-52.2)/10
#print('aaa=', a)

