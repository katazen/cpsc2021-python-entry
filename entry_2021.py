#-*- coding:UTF-8 -*-

import numpy as np, torch, os, json, wfdb, sys, bisect
from model import UNet
from utils import qrs_detect

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def find_closest_index(a, x):
    i = bisect.bisect_left(a, x)
    if i >= len(a):
        i = len(a) - 1
    elif i and a[i] - x > x - a[i - 1]:
        i = i - 1
    return i


def challenge_entry(sample_path):
    # 利用sig数据，返回列表，形式如[[s0,e0],[s1,e1],..,[sn,en]]
    sig, _, fs = load_data(sample_path)
    r_peaks = qrs_detect(sig[:,1], fs=200)
    # point_ans = []
    ans_01 = data_to_label(sig)
    # 判断三大类
    if sum(ans_01)/len(ans_01)<0.1:
        cls = 'Normal'
        r_ans = []
    elif sum(ans_01)/len(ans_01)>0.9:
        cls = 'AFf'
        # point_ans.append([0,len(ans_01)])
        r_ans = [[0,len(r_peaks)]]
    else:
        cls = 'AFp'
        r_ans = []
        # 如果为阵发性房颤，判断点位
        lst = []
        for i in range(len(ans_01)):
            cmp = ans_01[i-1] if i>0 else 0
            if ans_01[i]!=cmp:
                lst.append(i)
        s = lst[0::2]
        e = lst[1::2]
        if len(lst)%2 == 1:
            e.append(len(ans_01))
        s_ = s[:]
        e_ = e[:]
        if len(s) == 1:
            # point_ans.append([s[0], e[0]])
            r_ans.append([find_closest_index(r_peaks,s_[0]),find_closest_index(r_peaks,e_[0])])
        else:
            for j in range(len(s) - 1):
                if s[j + 1] - e[j] < 1000:
                    s_.remove(s[j + 1])
                    e_.remove(e[j])
        for k in range(len(s_)):
            r_ans.append([find_closest_index(r_peaks,s_[k]),find_closest_index(r_peaks,e_[k])])

    pred_dcit = {'predict_endpoints': r_ans}
    return pred_dcit

def save_dict(filename, dic):
    '''save dict into json file'''
    with open(filename,'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False)

def load_data(sample_path):
    sig, fields = wfdb.rdsamp(sample_path)
    length = len(sig)
    fs = fields['fs']
    return sig, length, fs

def data_to_label(data):
    # data:(ori_length,2)->x:(m,2,2000)->model_predict->y:(m,1,2000)->label:(ori_length,)
    length = data.shape[0]
    if length<2000:
        d = np.zeros((2000,2))
        d[-length:,:] = data
        data = d
        length = 2000
    data_num = length // 2000 if length % 2000 == 0 else length // 2000 + 1
    x = np.zeros((data_num, 2, 2000))
    for i in range(length // 2000):
        x[i, :, :] = data[2000 * i:2000 * i + 2000, :].T
    if length % 2000:
        x[-1, :, :] = data[-2000:,:].T
    y = np.zeros((data_num, 1, 2000))
    for k in range(5):
        model_name = 'model_train_I{}.pt'.format(k)
        label = model_pre(x,model_name)
        y+=label
    y = np.round(y/5).reshape(data_num,2000)
    label = np.zeros((length))
    for j in range(length // 2000):
        label[2000*j:2000 * j + 2000] = y[j]
    if length % 2000:
        label[-2000:]=y[-1]
    return label

def model_pre(X,model_name):
    net = UNet().to(device)
    checkpoint = torch.load(model_name)
    net.load_state_dict(checkpoint['model'])
    net.eval()
    x = torch.from_numpy(X)
    x = x.float().to(device)
    out = net(x).cpu().detach().numpy()
    return out

if __name__ == '__main__':
    DATA_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    # DATA_PATH = 'training_II'
    # RESULT_PATH = 'result'

    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    for i, sample in enumerate(test_set):
        print(sample)
        sample_path = os.path.join(DATA_PATH, sample)
        pred_dict = challenge_entry(sample_path)
        save_dict(os.path.join(RESULT_PATH, sample + '.json'), pred_dict)