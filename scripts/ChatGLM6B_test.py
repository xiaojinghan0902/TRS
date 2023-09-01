import json
from transformers import AutoTokenizer, AutoModel
import torch
import sys
import time
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    total_time = 0

    # 读取数据
    with open("../questions/test_basic_v2.txt", "r", encoding="utf-8") as f:
        data = f.readlines()
    f.close()

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained("../../SFT/ChatGLM6B/", trust_remote_code=True)
    model = AutoModel.from_pretrained("../../SFT/ChatGLM6B/", trust_remote_code=True).half().cuda()
    model.to(device)
    model = model.eval()

    # 开始获得结果
    result = []
    for inputtext_item in data:
        print("#################################################################")
        print(inputtext_item)
        print("><><><><><>")
        start_time = time.time()
        response, history = model.chat(tokenizer, str(inputtext_item), history=[])
        end_time = time.time()
        total_time += end_time - start_time
        print(response)
        result.append(response.replace("\n", "\\n")+"\n")
    
    print(total_time)
    
    # 写入文件
    with open("../modelanswers/ChatGLM6B_test_basic_v2.txt", "w", encoding="utf-8") as f:
        f.writelines(result)
    f.close()