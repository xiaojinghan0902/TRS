# coding=utf-8
import time
import json
import sys
import openai
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import xlwt


openai.api_type = "azure"
openai.api_base = "https://trschatsc.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "0b00962cd666462287f7baf216b2641f"

start = 0
end = 45
exp_times = 1

# 调用出错用try catch finally

def drawevalplot(scores):
    
    colors = ["#802A2A", "#B22222"]
    
    #误差线使用标准偏差SD 方差的平方根
    models = ["GPT4", "ChatGLM2"]
    
    # 数据处理
    means = []
    sds = []
    for model in scores.keys():
        temp = np.array(scores[model])
        means.append(np.mean(temp))
        sds.append(sqrt(np.var(temp)))
        
    index = np.arange(len(models))
    width = 0.4
    
    bar_scores = plt.bar(index, means, width, color=colors, yerr=sds)
    plt.xlabel('Models Name', fontsize=16)
    plt.ylabel('Scores',  fontsize=16)
    plt.title('The Assessment of LLMs',  fontsize=24)
    plt.xticks(index, models,  fontsize=12)

    for a,b in zip(index,means):   #柱子上的数字显示
        plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=17);

    # 创建图例
    plt.legend([bar_scores], ['Models'])
    plt.savefig("./chartllama.png")

if __name__ == '__main__':

    count = 0

    # 读取数据
    
    with open("finished\BasicskillPrompt.txt", "r", encoding="utf-8") as f:
        q_data = f.readlines()
    f.close()

    with open("BaichuanchatBasic.txt", "r", encoding="utf-8") as f:
        gpt4 = f.readlines()
    f.close()

    with open("finished\BasicskillAnswer.txt", "r", encoding="utf-8") as f:
        answer = f.readlines()
    f.close()
    
    model_answers = {"GPT4":gpt4}   
    
    
    p="你是一个打分员，需要参考正确答案对模型针对用户指令给出的回答进行打分。你需要对比正确答案与模型回答，来对模型的回答给出分数，满分为10分，最低分为0分。评价标准参考三个维度1.答案是否对用户有帮助(helpful)。2.答案是否真实(truthful)。3.答案是否是无害的(harmless)。“有帮助(helpful)”指的是输出应该符合用户的意图，帮助用户解决问题。以下是一些有帮助的行为示例：\n\n用清晰的语言进行写作。\n\n回答用户实际想要问的问题，即使他们表述错误。\n\n对国际性问题敏感（例如，“足球”不应该指美式足球，“总统”不一定指美国总统）。\n\n如果一条指令过于混乱，可以要求澄清并解释为什么指令会让人困惑。\n\n不要提供过于冗长或啰嗦的答案，也不要重复问题中已经提供的信息。\n\n不要假设超出已知信息的不相关上下文。例如，如果被要求“写一封礼貌的电子邮件回复该电子邮件：邮件内容”，输出不应该假设“我这次不能来，但下个周末有空”。但如果被要求“给苏格拉底写一封电子邮件”，则可以安全地假设输出可以编造上下文。\n\n请按照\"回答得分:\n原因:\n\"这样的形式输出分数。"

    rawmessages = [
        {"role": "system", "content": p},
    ]
    
    models = ["GPT4"]
    scores = {"GPT4":[]}
    
    print("开始打分")
    
    for j in range(exp_times):
        reviews = {"GPT4":[]}
        review_scores = {"GPT4":[]}
        for model in models:
            for i in range(start, end):
                if i in [6]:
                    continue
                messages = rawmessages[0:1]
                content = "请对比正确答案对模型回答的质量进行打分。\n\n问题：\n" + q_data[i].replace("\n","") + "\n\n答案：\n" + answer[i].replace("\n","") + "\n\n回答：\n" + model_answers[model][i].replace("\n","")
                messages.append({"role": "user", "content":content})
                while(True):
                    while(True):
                        flag = 0
                        try:
                            time.sleep(7) # openai连接次数限制
                            response = openai.ChatCompletion.create(
                                engine="gpt4",
                                messages=messages,
                                stream=False,
                                temperature=0.9,
                                max_tokens=1000,
                                top_p=1,
                                frequency_penalty=0,
                                presence_penalty=0.6
                            )
                        except BaseException as error:
                            print(error)
                            flag = 1
                        finally:
                            if flag == 0:
                                break
                    print(i)
                    print(response['choices'][0]['message']['content'])
                    if response['choices'][0]['message']['content'].split("\n")[0][5:].replace(" ","").isdigit() == True and response['choices'][0]['message']['content'].split("\n")[0][4] == ":":
                        break
        
                response = json.loads(str(response))
                re = response['choices'][0]['message']['content']
                reviews[model].append(re.replace("\n", "\\n")+"\n")
                review_scores[model].append(float(re.split("\n")[0].split(":")[1].replace(" ","")))


        # 将结果和评价保存为两个csv
        review = []

        for i in range(len(reviews["GPT4"])):
            review.append([reviews["GPT4"][i]])
        review = pd.DataFrame(columns=["GPT4回答评价"], data=review)
        review.to_csv("poem_review_GLM2_"+ str(j+1) +".xls", encoding="utf-8", header=1, index=1)
            
        # 计算每个模型的平均分
        for model in models:
            temp = np.array(review_scores[model])
            scores[model].append(np.mean(temp))
            
    # 计算总得分画图
    drawevalplot(scores)