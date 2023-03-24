import random
import pandas as pd
from collections import Counter
from numpy import array
from sklearn import tree, svm, linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

filepath = 'C:\\Users\\xiongqiang\\Desktop\\ssq.xlsx'
data = pd.read_excel(filepath)

col = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7']
data.columns = col
data = data.sort_index(ascending=False)

# 定义红球和蓝球的范围
red_range = range(1, 34)
blue_range = range(1, 17)


def rule_sequence():
    """
    顺序规则
    第一个数字关联后一个数字出现的概率
    """
    data_pre = data

    prob = {}
    for i in range(1,35):
        data_pre_latter = []
        for c in range(len(col)-2):
            data_pre_latter.extend(data_pre[data_pre[col[c]] == i][col[c+1]].values.tolist())

        prob[i] = count_(data_pre_latter)

    return prob


def count_(arr):
    # 计算每个数字出现的频率
    freq = Counter(arr)

    # 计算每个数字出现的概率
    total = len(arr)
    prob = {k: v / total for k, v in freq.items()}

    return prob


def rule_predict():
    """
    用以往的数字预测后面数字的出现概率
    n是样本量
    输出结果为
    每个位置的数字值和概率，总共7个位置
    """
    data_pre = data

    result = {}
    for i in range(1, 8):
        pred = []
        data_col = data_pre[col[i-1]].values.tolist()
        n_steps = 3
        x_train, y_train = split_sequence(data_col, n_steps)

        # 预测算法预测下一期可能出现的数字
        gnb = GaussianNB()
        y_predict = gnb.fit(x_train, y_train).predict([data_col[-n_steps:]])
        pred.extend(y_predict)

        logreg = linear_model.LogisticRegression(C=1, penalty="l1", tol=0.01, solver="saga")
        y_predict = logreg.fit(x_train, y_train).predict([data_col[-n_steps:]])
        pred.extend(y_predict)

        gpc = GaussianProcessClassifier()
        y_predict = gpc.fit(x_train, y_train).predict([data_col[-n_steps:]])
        pred.extend(y_predict)

        clf = svm.SVC(gamma='scale')
        y_predict = clf.fit(x_train, y_train).predict([data_col[-n_steps:]])
        pred.extend(y_predict)

        dtc = tree.DecisionTreeClassifier()
        y_predict = dtc.fit(x_train, y_train).predict([data_col[-n_steps:]])
        pred.extend(y_predict)
        #
        # lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        # y_predict = lda.fit(x_train, y_train).predict([data_col[-n_steps:]])
        # pred.extend(y_predict)

        # qda = QuadraticDiscriminantAnalysis()
        # y_predict = qda.fit(x_train, y_train).predict([data_col[-n_steps:]])
        # pred.extend(y_predict)

        rfc = RandomForestClassifier()
        y_predict = rfc.fit(x_train, y_train).predict([data_col[-n_steps:]])
        pred.extend(y_predict)

        ada = AdaBoostClassifier(n_estimators=100)
        y_predict = ada.fit(x_train, y_train).predict([data_col[-n_steps:]])
        pred.extend(y_predict)

        gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        y_predict = gbc.fit(x_train, y_train).predict([data_col[-n_steps:]])
        pred.extend(y_predict)

        # mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 2), random_state=1)
        # y_predict = mlp.fit(x_train, y_train).predict([data_col[-n_steps:]])
        # pred.extend(y_predict)

        # y_pred_gbc = gbc.fit(x_train, y_train).predict(x_train)
        # print("Number of mislabeled points out of a total %d points in gbc : %d" % (x_train.shape[0], (y_train != y_pred_gbc).sum()))

        result[i] = count_(pred)


    return result


def split_sequence(sequence, n_steps=3):
    """
    数据分割，三个数据为一组
    """
    X,y = list(),list()
    for i in range(len(sequence)):
        end_ix = i + n_steps            # 也就是y
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix],sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X),array(y)


def roulette(n_dict):
    """
    轮盘赌算法
    """
    if len(n_dict) > 1:
        # 计算每个对象的概率
        total_weight = sum(n_dict.values())
        probs = [v / total_weight for v in n_dict.values()]

        key_list = list(n_dict.keys())

        r = random.random()
        result = max(n_dict.items(), key=lambda x: x[1])[0]
        for i, p in enumerate(probs):
            if r < p:
                result = key_list[i]
                break
            r -= p

    elif len(n_dict) == 1:
        result = list(n_dict.keys())[0]
    else:
        result = 0
    return result


def new_predict(num=5):
    """
    默认产生5注最大可能的中奖号码
    """
    dict = rule_predict()

    # 增加随机蓝球
    blue_dict = dict[7]
    for i in range(5):
        blue_i = random.sample(blue_range, 1)[0]
        if blue_i not in blue_dict.keys():
            blue_dict[blue_i] = 0.1
        else:
            continue

    # 第一位到第三位的数字都用起来起来算起始数字
    red_list = []
    for i in range(num):
        red_list.append(roulette(dict[1]))

    red_dict = {key: value * 0.8 for key, value in dict[1].items()}
    red_dict_2 = {key: value * 0.2 for key, value in dict[2].items()}
    red_dict.update(red_dict_2)
    for i in range(num):
        red_list.append(roulette(red_dict))

    red_dict = {key: value * 0.6 for key, value in dict[1].items()}
    red_dict_2 = {key: value * 0.2 for key, value in dict[2].items()}
    red_dict_3 = {key: value * 0.2 for key, value in dict[3].items()}
    red_dict.update(red_dict_2)
    red_dict.update(red_dict_3)
    for i in range(num):
        red_list.append(roulette(red_dict))

    seq = rule_sequence()
    result = []
    for code in red_list:
        result_code = []
        first_code = code
        for i in range(1, 7):
            result_code.append(first_code)
            try:
                code_ = seq[first_code]
                first_code = roulette(code_)
            except:
                first_code = random.sample(red_range, 1)[0]

        result.append(result_code)

    res = []
    length = len(result)

    # 打乱顺序，生成指定的注数
    random.shuffle(result)

    j = 0
    for i in range(length):
        blue = roulette(dict[7])
        red = result[i]

        if blue not in red and len(set(red)) == 6 and 0 not in red:
            res.append([red, [blue]])
            j += 1

            if j == num:
                break
        else:
            continue

    return res

print(new_predict(num=5))
