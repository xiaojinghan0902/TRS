import numpy as np
import matplotlib.pyplot as plt

# 平均房价, 波动范围
scores = [82, 47, 18, 14, 6]
tolerance_a = [5, 4, 3.5, 3, 2]
# tolerance_b = (3, 2, 2, 2.5, 1)

index = np.arange(len(scores))
width = 0.4
models = ['GPT4', 'Ours', 'PCLUE770M', 'PanGu2.6B', 'LLaMA7B']

bar_scores = plt.bar(index, scores, width, color='#02ccfe', yerr=tolerance_a)
plt.xlabel('Model Name', fontsize=16)
plt.ylabel('Scores',  fontsize=16)
plt.title('The Assessment of LLM',  fontsize=24)
plt.xticks(index, models,  fontsize=12)

for a,b in zip(index,scores):   #柱子上的数字显示
 plt.text(a,b,'%d'%b,ha='center',va='bottom',fontsize=17);

# 创建图例
plt.legend([bar_scores], ['Models'])
plt.savefig("./chart.png")