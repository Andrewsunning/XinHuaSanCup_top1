import jieba
from sklearn.metrics import f1_score

'''
1. get_label，多标签转码，获取犯罪名称到犯罪代码的映射，return 字符串
2. get_words，分词，把句子中的词提取出来，return 字符串
3. f1_avg_scorer，评估指标，return 浮点数
4. get_accusation，多标签解码，return 字符串
'''

# 多标签转码
def get_label(str_accusations, dic_labels):
    lst_accusations = str_accusations.split(';')
    lst_labels = []
    for accusation in lst_accusations:
        lst_labels.append(dic_labels[accusation])
    return ','.join(lst_labels)

# 分词
def get_words(str_sentence, lst_stopwords):
    return ' '.join(list(set(jieba.lcut(str_sentence, cut_all=False)) - set(lst_stopwords)))

# 评估指标
def f1_avg_scorer(y_true, y_pred):
    f1_avg_score = (f1_score(y_true, y_pred, average='macro') + f1_score(y_true, y_pred, average='micro')) / 2
    return f1_avg_score

# 获取指控信息
def get_accusation(labels, lst_labels, dic_labels_r):
    labels = labels.split(':')[1:]
    accusations = ''
    for i in range(len(lst_labels)):
        if labels[i] == '1':
            accusations += ';' + dic_labels_r[lst_labels[i]]
    return accusations[1:]