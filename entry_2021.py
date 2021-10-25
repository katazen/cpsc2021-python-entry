import os, wfdb, numpy as np, torch, itertools, json, sys
from model import UNet


class load_model:

    def __init__(self):
        # 初始化函数，加载模型
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.net0 = UNet().to(self.device)
        self.checkpoint0 = torch.load('model_train1018_{}_withP_{}.pt'.format('I', 0))
        self.net0.load_state_dict(self.checkpoint0['model'])
        self.net0.eval()
        self.net1 = UNet().to(self.device)
        self.checkpoint1 = torch.load('model_train1018_{}_withP_{}.pt'.format('II', 0))
        self.net1.load_state_dict(self.checkpoint1['model'])
        self.net1.eval()
        self.net2 = UNet().to(self.device)
        self.checkpoint2 = torch.load('model_train1018_{}_withP_{}.pt'.format('II', 1))
        self.net2.load_state_dict(self.checkpoint2['model'])
        self.net2.eval()

    def predict(self, data, length):
        # 输入(n,2,2000)的测试数据，返回(length,)的标签
        data = torch.from_numpy(data)
        data = data.float().to(self.device)
        out = (self.net0(data)+self.net1(data)+self.net2(data))/3
        out = np.argmax(out.cpu().detach().numpy(),axis=1)

        # out0 = self.net0(data).cpu().detach().numpy()
        # out1 = self.net1(data).cpu().detach().numpy()
        # out2 = self.net2(data).cpu().detach().numpy()
        # out = (out0+out1+out2)/3
        # out = np.argmax(out, axis=1)
        dim = out.shape[0]
        if dim == 1:
            ans = out[0, :length]
        elif dim == 2:
            ans = np.concatenate((out[0, :1900], out[1, 1900 - length:]))
        else:
            ans = out[0, :1900]
            for i in range(1, dim - 1):
                ans = np.concatenate((ans, out[i, 100:-100]))
            ans = np.concatenate((ans, out[-1, dim * 1800 - length - 1700:]))
        return ans


def data_val(SAMPLE_PATH):
    # 输入文件路径，返回(n,2,2000)的测试数据
    rec = wfdb.rdrecord(SAMPLE_PATH)
    sig = rec.p_signal.T
    length = sig.shape[1]
    if length < 2000:
        data = np.zeros((1, 2, 2000))
        data[0, :, :length] = sig
    else:
        dim0 = (length - 200) // 1800 if (length - 200) % 1800 == 0 else (length - 200) // 1800 + 1
        data = np.zeros((dim0, 2, 2000))
        for i in range((length - 200) // 1800):
            data[i, :, :] = sig[:, 1800 * i:1800 * i + 2000]
        if (length - 200) % 1800:
            data[-1, :, :] = sig[:, -2000:]
    return data, length


def get_result(ANS_DATA):
    # 输入(length,)的标签，返回起止点
    # 将预测标签转换为R波所在位置列表r_pos和其对应的R波种类列表note
    t = [(k, sum(1 for _ in vs)) for k, vs in itertools.groupby(ANS_DATA)]
    note = []
    r_pos = []
    lens = 0
    for item in t:
        lens += item[1]
        if item[0] == 1 and item[1] >= 40:
            note.append(1)
            r_pos.append(lens - int(item[1] * 0.7))
        elif item[0] == 2 and item[1] >= 40:
            note.append(2)
            r_pos.append(lens - int(item[1] * 0.7))
    # 根据note判断数据类型，并且根据r_pos返回相应结论
    se = [(k, sum(1 for _ in vs)) for k, vs in itertools.groupby(note)]
    r = 0
    st_ed = []
    for elem in se:
        r += elem[1]
        if elem[0] == 2:
            st_ed.append([r_pos[r - elem[1]], r_pos[r - 1]])
    # st_ed = st_ed_check(st_ed)
    pred_dcit = {'predict_endpoints': st_ed}
    return pred_dcit


def save_dict(filename, dic):
    '''save dict into json file'''
    with open(filename, 'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False)


def st_ed_check(ST_ED):
    # 对最终结论的列表添加筛选规则
    pass


if __name__ == '__main__':
    DATA_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    # DATA_PATH = 'training_II'
    # RESULT_PATH = 'result'

    m = load_model()

    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    for i, sample in enumerate(test_set):
        print(sample)
        sample_path = os.path.join(DATA_PATH, sample)
        train_data, data_length = data_val(sample_path)
        ans_data = m.predict(train_data, data_length)
        ans = get_result(ans_data)
        save_dict(os.path.join(RESULT_PATH, sample + '.json'), ans)
