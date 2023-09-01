import os, time, json, traceback
import sys
import openai
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://trschatsc.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_KEY"] = "0b00962cd666462287f7baf216b2641f"

openai.api_type = "azure"
openai.api_base = "https://trschatsc.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "0b00962cd666462287f7baf216b2641f"

if __name__ == '__main__':
    # 读取数据
    
    
    with open('logicprompt.txt', "r", encoding="utf-8") as f:
        q_data = f.readlines()
    f.close()
    
    result = []
    
    prompt = """
        下面是一些用户问题，请用中文回答它们。
    """

    
    for i in range(len(q_data)):
        messages = [
            {"role": "system", "content": prompt},
        ]
        messages.append({"role": "user", "content":q_data[i]})
        
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
        
        response = json.loads(str(response))
        re = response['choices'][0]['message']['content']
        result.append(re.replace("\n", "\\n")+"\n")
        
        print(i)
    
    # 写入文件
    with open("gpt4_test_basicV3.txt", "w", encoding="utf-8") as f:
        f.writelines(result)
    f.close()