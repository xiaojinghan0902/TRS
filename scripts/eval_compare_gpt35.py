import json
import sys
import openai

openai.api_type = "azure"
openai.api_base = "https://trschatsc.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "0b00962cd666462287f7baf216b2641f"

if __name__ == '__main__':
    # 读取数据
    
    with open("test_basic.txt", "r", encoding="utf-8") as f:
        q_data = f.readlines()
    f.close()
    
    with open("result_test_basic.txt", "r", encoding="utf-8") as f:
        res_data = f.readlines()
    f.close()
    
    with open("baseline_test_basic.txt", "r", encoding="utf-8") as f:
        base_data = f.readlines()
    f.close()
    
    result = []
    
    prompt = """
        你是一个打分员，需要对两个模型针对用户指令给出的回答进行打分。你需要对比两个输出语句，来分别对模型的回答给出分数，满分为1分，最低分为0分。请按照\"回答1得分:\n回答2得分:\n原因:\n\"这样的形式输出分数。评价标准要求回答的结果越合理越友善越有意义则得分越高。\n\n
    """
    
    for i in range(len(q_data)):
        messages = [
            {"role": "system", "content": prompt},
        ]
        # 给两个例子
        exmcontent1 = """请对比下面两个模型回答的质量进行打分。\n\n问题：\n从此文本中提取机场代码： “我想从奥兰多飞往波士顿\n\n回答1：\n从这句话中，我们无法直接提取机场代码。不过，我可以提供这两个城市的常用机场代码：\\n奥兰多（Orlando）：MCO（奥兰多国际机场）\\n波士顿（Boston）：BOS（洛根国际机场）\n\n回答2：\n“我想从奥兰多飞往波士顿”，可以提取出以下机场代码：\\nORD（奥兰多国际机场）\\nBOS（波士顿布莱恩斯国际机场）"""
        exmresponse1 = """回答1得分:0.8\n回答2得分:0.5\n原因:由于两个回答都理解了用户问题因此给出基础分0.3分，从回答质量上看回答1的回答更有意义并且更合理所以给出0.8分；回答2不够友善并且给出错误机场代码所以给出0.5分"""
        messages.append({"role": "user", "content":exmcontent1})
        messages.append({"role": "assistant", "content":exmresponse1})
        
        exmcontent2 = """请对比下面两个模型回答的质量进行打分。\n\n问题：\n根据描述帮我起个产品名字吧\\n产品描述：家用搅拌机\\n关键词：快速、健康、小巧\n\n回答1：\n产品名：健康快速搅拌机 "健速宝"\n\n回答2：\n针对您的需求，我推荐以下几个产品名称：\\n快立方（QuickCup）：这个名字结合了“快速”和“果汁搅拌机”的含义，简洁易记，同时也符合小巧、便于携带的产品特点。\\n健康先锋（Health Pioneers）：这个名称强调了产品的健康属性，突出了其快速搅拌、均匀混合的特点，同时也有一定的品牌识别度。\\n果蔬轻骑兵（Fruit & Veggie Light Cavalry）：这个名字结合了“果蔬”和“轻骑兵”的含义，形象生动，突出了产品的快速搅拌、轻便易用的特点。\\n希望以上几个名称能够给您提供一些参考和帮助，祝您的产品取得成功！"""
        exmresponse2 = """回答1得分:0.6\n回答2得分:0.9\n原因:由于两个回答都理解了用户问题因此给出基础分0.3分，从回答质量上看回答1的回答符合问题要求但是没有回答2的丰富所以给分0.6分；回答2不仅给出了合理的回答并且给出了很多选项，回答更有意义也更合理，所以给分0.9分"""
        messages.append({"role": "user", "content":exmcontent2})
        messages.append({"role": "assistant", "content":exmresponse2})
        
        content = """请对比下面两个模型回答的质量进行打分。\n\n问题：\n""" + q_data[i].replace("\n","") + """\n\n回答1：\n""" + res_data[i].replace("\n","") + """\n\n回答2：\n""" + base_data[i].replace("\n","")
        messages.append({"role": "user", "content":content})
        
        response = openai.ChatCompletion.create(
            engine="gpt-turbo",
            messages=messages,
            stream=False,
            temperature=0.9,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6
        )
        
        response = json.loads(str(response))
        re = response['choices'][0]['message']['content']
        result.append(re+"\n")
        
        print(i)
    
    # 写入文件
    with open("eval_compare_gpt35.txt", "w", encoding="utf-8") as f:
        f.writelines(result)
    f.close()