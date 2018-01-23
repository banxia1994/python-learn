# coding:utf-8

import copy
from math import log2
#获取信息熵
def get_shanno_entropy(self,values):
    uniq_vals = set(values)
    val_nums = {key:values.count(key) for key in uniq_vals}
    probs = [v/len(values) for k,v in val_nums.items()]
    entropy = sum([-prob*log2(prob) for prob in probs])
    return entropy