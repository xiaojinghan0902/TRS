from base_llm import BaseLLM
import time

if __name__ == '__main__':
    llm_model_dict = {"chatglm2_6b": f"http://1.13.24.50:7103/llm/chatglm2_6b",
                      "baichuan":f"http://1.13.24.50:7102/llm/baichuanchat-13b"}
    with open('ChuiZhi.txt', "r", encoding="utf-8") as f:
        q_data = f.readlines()
    f.close()
    #q_data = ['北京都有哪些美食']
    for i in range(len(q_data)):
        responses = BaseLLM(model_name='baichuan').generatorAnswer(prompt=q_data[i], llm_model_dict=llm_model_dict, streaming=False)
        for resp in responses:
            break
        time.sleep(1)
        # 将答案连接成一个字符串  
        answer_str = ''.join(resp.llm_output['answer'].split('\n'))  
        # 将答案写入 output.xlsx 文件  
        with open('outputB.xlsx', 'a') as f:  
            print(answer_str, file=f)  
        print(i)
        print(answer_str)
   
    
