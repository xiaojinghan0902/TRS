import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

def drawevalplot(scores):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    colors = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#00FFFF", "#0000FF", "#8B00FF", "#00000F", "#F00000"]

    # 误差线使用标准偏差SD 方差的平方根
    models = ["GPT4", "glm130", "glm6", "wxyy", "tyqy", "minimax", "xfxh"]
    qlabels = ['开放问答', '代码', '文字生成', '封闭问答', '脑筋急转弯', '分类', '提取', '重写', '翻译']

    # 数据处理
    qa_label_result_means = {}
    qa_label_result_sds = {}
    for model in scores.keys():
        for label in scores[model].keys():
            if label not in qa_label_result_means.keys():
                qa_label_result_means[label] = []
            if label not in qa_label_result_sds.keys():
                qa_label_result_sds[label] = []
            temp = np.array(scores[model][label])
            qa_label_result_means[label].append(np.mean(temp))
            qa_label_result_sds[label].append(sqrt(np.var(temp)))

    index = np.array(range(len(models)))
    width = 0.09

    plt.xlabel('Models Name', fontsize=16)
    plt.ylabel('Scores', fontsize=16)
    plt.title('The Assessment of LLMs', fontsize=24)
    plt.xticks(range(len(models)), models, fontsize=12)
    count = 0
    for key in qa_label_result_means.keys():
        plt.bar(index + count*width, qa_label_result_means[key], color=colors[count], width=width)
        for a, b in zip(range(len(models)), qa_label_result_means[key]):  # 柱子上的数字显示
            plt.text(a + count*width, b, '%.1f' % b, ha='center', va='bottom', fontsize=6)
        count += 1

    # 创建图例
    plt.legend(qlabels)
    plt.show()
    #plt.savefig("./chart0612.png")


labels = ['开放问答', '代码', '文字生成', '封闭问答', '脑筋急转弯', '分类', '提取', '重写', '翻译']
resultIndex = {}

qa_datas = pd.read_csv("./review/datasets.csv").values.tolist()
for i in range(62):
    if i in [6]:
        continue
    if i < 6:
        index = i
    else:
        index = i-1
    label = qa_datas[i][8]
    if label not in resultIndex.keys():
        resultIndex[label] = []
        resultIndex[label].append(index)
    else:
        resultIndex[label].append(index)

models = ["GPT4", "ChatGLM130B", "ChatGLM6B", "文心一言", "通义千言", "MiniMax", "讯飞星火"]
reviewScores = {"GPT4": {}, "ChatGLM130B": {}, "ChatGLM6B": {}, "文心一言": {}, "通义千言": {}, "MiniMax": {}, "讯飞星火": {}}
modelReview1 = pd.read_csv("./review/magua_base_generation_test_review_v2_1.csv").values.tolist()
modelReview2 = pd.read_csv("./review/magua_base_generation_test_review_v2_2.csv").values.tolist()

for j in range(len(modelReview1)):
    qa_key = ""
    for key in resultIndex.keys():
        if j in resultIndex[key]:
            qa_key = key
            break

    for i in range(len(modelReview1[j])):
        if i == 0:
            continue
        temp = modelReview1[j][i].split("\\n")
        score = int(temp[0][5:].replace(" ",""))
        if qa_key not in  reviewScores[models[i-1]]:
            reviewScores[models[i - 1]][qa_key] = [[], []]
            reviewScores[models[i - 1]][qa_key][0].append(score)
        else:
            reviewScores[models[i - 1]][qa_key][0].append(score)

for j in range(len(modelReview2)):
    qa_key = ""
    for key in resultIndex.keys():
        if j in resultIndex[key]:
            qa_key = key
            break

    for i in range(len(modelReview2[j])):
        if i == 0:
            continue
        temp = modelReview2[j][i].split("\\n")
        score = int(temp[0][6:])
        reviewScores[models[i - 1]][qa_key][1].append(score)

score = {"GPT4": {}, "ChatGLM130B": {}, "ChatGLM6B": {}, "文心一言": {}, "通义千言": {}, "MiniMax": {}, "讯飞星火": {}}
for modelkey in score.keys():
    reviewS4M = reviewScores[modelkey]
    for label in reviewS4M.keys():
        if label not in score[modelkey]:
            score[modelkey][label] = []
        for sc_list in reviewS4M[label]:
            sc_list = np.array(sc_list)
            score[modelkey][label].append(np.mean(sc_list))

drawevalplot(score)


