from answer_result import AnswerResult
from typing import Optional, List
import requests,time, traceback
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import List, Optional
from ast import literal_eval
import re, pdb

class BaseLLM(LLM):
    model_name: str
    history_len: int
    
    def __init__(self, model_name: str = 'chatglm2_6b', history_len: int = 0):
        super().__init__(model_name=model_name, history_len=history_len)
        self.model_name = model_name
        self.history_len = history_len

    def resolve_json(self, resp: str) -> str:
        if self.model_name in ['zh_alpaca13b_llamacpp', 'chatglm', 'chatglm2_6b']:
            pattern = r'\{.*\}'
            resp = resp.decode('utf8')
            resp = re.search(pattern=pattern, string=resp)
            if resp:
                resp = literal_eval(resp.group())
            else:
                return {"code": 204}

            return resp
        else:
            return {"code": 500, "result": "模型回复发生了一点小问题，请稍后再试！"}

    def llm_req(self, url: str, prompt: str, history: List[List], temperature: float, streaming: bool):
        head = {"Content-Type": "application/json; charset=UTF-8", 'Connection': 'close'}
        try:
            history = []
            res = requests.request(method='POST', 
                                   url=url, 
                                   headers=head,
                                   json={"query": prompt, 
                                         "background_info": "",
                                         "histories": [], 
                                         "temperature": temperature, 
                                         "stream": streaming,
                                         "use_type": "chat"},
                                   stream=True)
            if res.status_code != 200:
                answer_result = AnswerResult()
                output_text = "返回出错，请重新提问！"
                if streaming:
                    answer_result.llm_output = {"answer": output_text, "finish": True}
                else:
                    history.append(["", output_text])
                    answer_result.llm_output = {"answer": output_text}
                answer_result.history = history
                yield answer_result
            else:
                if streaming:
                    for r1 in res:
                        answer_result = AnswerResult()
                        answer_result.history = history
                        token = self.resolve_json(resp=r1)
                        if token['code'] == 500:
                            answer_result.llm_output = {"answer": '返回出错，请重新提问！', 'error_msg': token['result'], "finish": True}
                            yield answer_result
                        elif token['code'] == 204:  # ping了这台服务器，无返回结果 
                            continue
                        else:
                            if token['finish'] == 'stop':
                                answer_result.llm_output = {"answer": '', "finish": True}
                                yield answer_result
                            else:
                                answer_result.llm_output = {"answer": token['content'], "finish": False}
                                yield answer_result
                else:
                    output_text = res.json()['content']
                    history.append(["", output_text])
                    answer_result = AnswerResult()
                    answer_result.history = history
                    answer_result.llm_output = {"answer": output_text}
                    yield answer_result

        except Exception as e:
            info = traceback.format_exc()
            answer_result = AnswerResult()
            output_text = "返回出错，请重新提问！"
            if streaming:
                answer_result.llm_output = {"answer": output_text, "finish": True}
            else:
                history.append(["", output_text])
                answer_result.llm_output = {"answer": output_text}
            answer_result.history = history
            yield answer_result


    def generatorAnswer(self, prompt: str,
                        llm_model_dict: dict,
                        historys: List[List[str]] = [],
                        streaming: bool = False,
                        temperature = 0.5,
                        **kwargs):
        for resp in self.llm_req(llm_model_dict[self.model_name], prompt, historys, temperature, streaming):
            yield resp


    @property
    def _llm_type(self) -> str:
        return self.model_name

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        for response in self.generatorAnswer(prompt=prompt, streaming=False):
            break
        res = response.llm_output['answer']
        return res
