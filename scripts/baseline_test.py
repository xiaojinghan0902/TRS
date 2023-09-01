import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
import torch
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess(text):
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text

def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t")

def answer(text, sample=True, top_p=1, temperature=0.7):
    '''sample：是否抽样。生成任务，可以设置为True;
    top_p：0-1之间，生成的内容越多样'''
    text = preprocess(text)
    encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(device)
    if not sample:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, num_beams=1, length_penalty=0.6)
    else:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, do_sample=True, top_p=top_p, temperature=temperature, no_repeat_ngram_size=3)
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0])
    print("end...")

if __name__ == '__main__':
    # 读取数据
    with open("test_basic.txt", "r", encoding="utf-8") as f:
        data = f.readlines()
    f.close()

    # 加载模型
    tokenizer = T5Tokenizer.from_pretrained("ClueAI/PromptCLUE-base")
    model = T5ForConditionalGeneration.from_pretrained("ClueAI/PromptCLUE-base")
    model.to(device)

    # 开始获得结果
    result = []
    for inputtext_item in data:
        print("#################################################################")
        print(inputtext_item)
        print("><><><><><>")
        output_text = answer(inputtext_item)
        output_text = output_text.replace("\n", "\\n").replace("\t", "\\t")
        print(output_text)
        result.append(output_text+"\n")

    # 写入文件
    with open("baseline_test_basic.txt", "w", encoding="utf-8") as f:
        f.writelines(result)
    f.close()