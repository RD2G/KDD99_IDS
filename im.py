'''
《KDDCup99入侵检测实验数据的标识类型》
Normal	正常记录	normal
DOS	拒绝服务攻击	back, land, neptune, pod, smurf, teardrop
Probing	监视和其他探测活动	ipsweep, nmap, portsweep,satan
R2L	来自远程机器的非法访问	ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster
U2R	普通用户对本地超级用户特权的非法访问	buffer_overflow, loadmodule, perl, rootkit

《41个固定的特征属性》
duration,protocol_type,service,flag,src_bytes,dst_bytes,land,
wrong_fragment,urgent,ho,num_failed_logins,logged_in,num_compromised,
root_shell,su_attempted,num_root,num_file_creations,num_shells,
num_access_files,num_outbound_cmds,is_host_login,is_guest_login,count,
srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,
same_srv_rate,diff_srv_rate,srv_diff_host_rate,dst_host_count,
dst_host_srv_count,dst_host_same_srv_rate,dst_host_diff_srv_rate,
dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,
dst_host_serror_rate,dst_host_srv_serror_rate,dst_host_rerror_rate,
dst_host_srv_rerror_rate,class
比较能体现出状态变化的是前31个特征属性，其中9个离散型，22个连续型

单个TCP连接的基本特征:
Duration	连接时间长度（单位：秒）	连续型
Protocol_type	协议类型，如tcp,udp	离散型
Service	在目标机的网络服务，如http,telnet等	离散型
src_bytes	源地址到目标地址的数据流量	连续型
dst_bytes	目标地址到源地址的数据流量	连续型
flag	连接状态（正常或错误）	离散型
land	1表示数据连接源地址和目标地址为同一主机或端口；0表示其他	离散型
wrong_fragment	错误碎片的数目	连续型
urgent	紧迫数据包的个数	连续型

一次连接中包含的内容特征:
hot	访问系统敏感文件和目录的次数	连续型
mum_failed_logins	尝试登录失败的次数	连续型
loggged_in	1表示成功登录，0表示其他	离散型
num_compromised	受到威胁状态的次数	连续型
root_shell	1表示超级用户的shell外壳，0表示其他	离散型
su_attempted	1表示命令执行尝试，0表示其他	离散型
num_root	root权限访问的次数	连续型
num_file_creations	文件创作的操作次数	连续型
num_shells	shell提示符合的个数	连续型
num_access_files	访问控制文件的次数	连续型
num_outbound_cmds	一次ftp会话中传递命令的次数	连续型
is_hot_login	1表示属于热点清单的登录，0表示其他	离散型
is_guest_login	1表示guest用户登录，0表示其他用户名登录	离散型
一般使用 KDDCup99 中的网络入侵检测数据包kddcup_data_10percent。
kddcup_data_10percent数据包是对kddcup_data数据包(约490万条数据记录) 10%的抽样
'''

import os
import sys
import time
import csv
import re
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier



# tstpth = './kddcup.data.corrected'  #完整测试集
# trnpth = './kddcup.data_10_percent_corrected'   #训练集
# trndpth = './trained_data.csv'   #训练结果
# trndpth1 = './trained_data1.csv'    #标准化结果
# trndpth2 = './trained_data2.csv'    #归一化结果
protocols = ['icmp', 'tcp', 'udp']
services = ['IRC', 'X11', 'Z39_50', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain',
            'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher',
            'hostnames', 'http', 'http_443', 'icmp', 'imap4', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link',
            'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp',
            'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje',
            'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i',
            'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois']
flags = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
train_label_types = ['back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.', 'ipsweep.', 'land.',
                    'loadmodule.', 'multihop.', 'neptune.', 'nmap.', 'normal.', 'perl.', 'phf.', 'pod.',
                    'portsweep.', 'rootkit.', 'satan.', 'smurf.', 'spy.', 'teardrop.', 'warezclient.',
                    'warezmaster.']
test_label_types = ['apache2.', 'back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'httptunnel.', 'imap.',
                    'ipsweep.', 'land.', 'loadmodule.', 'mailbomb.', 'mscan.', 'multihop.', 'named.', 'neptune.',
                    'nmap.', 'normal.', 'perl.', 'phf.', 'pod.', 'portsweep.', 'processtable.', 'ps.', 'rootkit.',
                    'saint.', 'satan.', 'sendmail.', 'smurf.', 'snmpgetattack.', 'snmpguess.', 'sqlattack.',
                    'teardrop.', 'udpstorm.', 'warezmaster.', 'worm.', 'xlock.', 'xsnoop.', 'xterm.']
label_types = [['normal.'],
              ['ipsweep.', 'mscan.', 'nmap.', 'portsweep.', 'saint.', 'satan.'],
              ['apache2.', 'back.', 'land.', 'mailbomb.', 'neptune.', 'pod.', 'processtable.', 'smurf.', 'teardrop.', 'udpstorm.'],
              ['buffer_overflow.', 'httptunnel.', 'loadmodule.', 'perl.', 'ps.', 'rootkit.', 'sqlattack.', 'xterm.'],
              ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'named.', 'phf.', 'sendmail.', 'snmpgetattack.',
              'snmpguess.', 'spy.', 'warezclient.', 'warezmaster.', 'worm.', 'xlock.', 'xsnoop.']]


# 数据预处理 字符转化数值
def String2Value(sr,ds):
    try:
        with open(sr) as f:
            csv_datas = csv.reader(f)
            with open(ds,'w',newline='') as t:
                cw = csv.writer(t)
                for row in csv_datas:
                    line = np.array(row)  # 将每行数据存入line数组里
                    # print(line)
                    line[1] = protocols.index(line[1])  # 将源文件行中3种协议类型转换成数字标识
                    line[2] = services.index(line[2])  # 将源文件行中70种网络服务类型转换成数字标识
                    line[3] = flags.index(line[3])  # 将源文件行中11种网络连接状态转换成数字标识
                    for label in label_types:
                        if label.count(line[41]) > 0:
                            line[41] = label_types.index(label) # 将源文件行中23种攻击类型转换成数字标识
                        # else:
                        #     label_types.append(input[41])
                        #     line[41] = label_types.index(label)
                    cw.writerow(line)
                    print(line)
            # print(csv_datas)
    except Exception:
        print('打开kddcup99训练集失败')

# def StopWord():
#     pass

#标准化处理
def Standardization(sr,ds):
    datas = pd.read_csv(sr)
    x = preprocessing.scale(datas)
    with open(ds,'w',newline='') as f:
        # print(datas)
        cw = csv.writer(f)
        for row in x:
            cw.writerow(row)
    # print(pd.read_csv(trndpth))


#归一化处理，将数据缩至0-1之间，采用MinMaxScaler函数
def Normalization(sr,ds):
    datas = pd.read_csv(sr)
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(datas)
    # print(x)
    with open(ds,'w',newline='') as f:
        # print(datas)
        cw = csv.writer(f)
        for row in x:
            cw.writerow(row)

#特征选取
def Feature(X, y):
    model = ExtraTreesClassifier()
    model.fit(X, y) #X作为特征向量，y作为目标变量
    # display the relative importance of each attribute
    print(model.feature_importances_)



# if __name__ == '__main__':
    # tstpth = './kddcup.data.corrected'  # 完整测试集
    # tstpth10p = './corrected'   #10%测试集
    # tstdpth = './tested_data.csv'
    # tstdpth1 = './tested_data1.csv'  # 标准化结果
    # tstdpth2 = './tested_data2.csv'  # 归一化结果
    #
    # protocols = ['icmp', 'tcp', 'udp']
    # services = ['IRC', 'X11', 'Z39_50', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain',
    #             'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher',
    #             'hostnames', 'http', 'http_443', 'icmp', 'imap4', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link',
    #             'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp',
    #             'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje',
    #             'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i',
    #             'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois']
    # flags = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
    # train_label_types = ['back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.', 'ipsweep.', 'land.',
    #                      'loadmodule.', 'multihop.', 'neptune.', 'nmap.', 'normal.', 'perl.', 'phf.', 'pod.',
    #                      'portsweep.', 'rootkit.', 'satan.', 'smurf.', 'spy.', 'teardrop.', 'warezclient.',
    #                      'warezmaster.']
    # test_label_types = ['apache2.', 'back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'httptunnel.', 'imap.',
    #                     'ipsweep.', 'land.', 'loadmodule.', 'mailbomb.', 'mscan.', 'multihop.', 'named.', 'neptune.',
    #                     'nmap.', 'normal.', 'perl.', 'phf.', 'pod.', 'portsweep.', 'processtable.', 'ps.', 'rootkit.',
    #                     'saint.', 'satan.', 'sendmail.', 'smurf.', 'snmpgetattack.', 'snmpguess.', 'sqlattack.',
    #                     'teardrop.', 'udpstorm.', 'warezmaster.', 'worm.', 'xlock.', 'xsnoop.', 'xterm.']
    # label_types = [['normal.'],
    #                ['ipsweep.', 'mscan.', 'nmap.', 'portsweep.', 'saint.', 'satan.'],
    #                ['apache2.', 'back.', 'land.', 'mailbomb.', 'neptune.', 'pod.', 'processtable.', 'smurf.',
    #                 'teardrop.', 'udpstorm.'],
    #                ['buffer_overflow.', 'httptunnel.', 'loadmodule.', 'perl.', 'ps.', 'rootkit.', 'sqlattack.',
    #                 'xterm.'],
    #                ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'named.', 'phf.', 'sendmail.',
    #                 'snmpgetattack.',
    #                 'snmpguess.', 'spy.', 'warezclient.', 'warezmaster.', 'worm.', 'xlock.', 'xsnoop.']]
    #
    #
    # trnpth = './kddcup.data_10_percent_corrected'  # 训练集
    # trndpth = './trained_data.csv'  # 训练结果
    # trndpth1 = './trained_data1.csv'  # 标准化结果
    # trndpth2 = './trained_data2.csv'  # 归一化结果
    #
    #
    # String2Value(tstpth10p,tstdpth)
    # Standardization(tstdpth,tstdpth1)
    # Normalization(tstdpth,tstdpth2)