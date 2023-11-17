import time
import logging
import requests
from typing import Optional, List, Dict, Mapping, Any

import langchain
from langchain.llms.base import LLM

###################说明############################
#这只是一个tigerbot的云端api如何在langchian中替代openai来处理
#问答的示例，在具体的项目中你可能需要自己根据实际情况进行调整。

logging.basicConfig(level=logging.INFO)
#这里的apikey就是你在tigerbot上申请到的apikey
API_KEY=''

class Tigerbot_chatapi(LLM):
    # 模型服务url
    url = "https://api.tigerbot.com/bot-service/ai_service/gpt"
    modelVersion="tigerbot-7b-sft"
    #选择使用的模型

    @property
    def _llm_type(self) -> str:
        return "Tigerbot"

    def _construct_query(self, prompt: str) -> Dict:
        """构造请求体
        """
        payload = {
        "text": prompt,
        "modelVersion":self.modelVersion 
        }        
        return payload

    @classmethod
    def _post(cls, url: str,
        query: Dict) -> Any:
        """POST请求
        """
        _headers = {"Content_Type": "application/json",'Authorization':'Bearer ' + API_KEY}
        with requests.session() as sess:
            resp = sess.post(url, 
                json=query, 
                headers=_headers, 
                timeout=60)#时间可以根据具体需求进行调整。
        return resp

    
    def _call(self, prompt: str, 
        stop: Optional[List[str]] = None) -> str:
        """_call
        """
        # construct query
        query = self._construct_query(prompt=prompt)

        # post
        resp = self._post(url=self.url,
            query=query)
        
        if resp.status_code == 200:
            resp_json = resp.json()
            predictions = resp_json["data"]["result"][0]
            return predictions
        else:
            resp_json = resp.json()
            errmsg = resp_json["msg"]
            return errmsg
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters.
        """
        _param_dict = {
            "url": self.url
        }
        return _param_dict

if __name__ == "__main__":
    llm = Tigerbot_chatapi()
    #这里的llm即可对应langchain很多方法中需要调用模型的llm变量类型。下面是一个简单的控制台问答过程。
    while True:
        user_input = input("user: ")

        begin_time = time.time() * 1000
        # 请求模型
        response = llm(user_input)
        end_time = time.time() * 1000
        used_time = round(end_time - begin_time, 3)
        logging.info(f"Tigerbot process time: {used_time}ms")

        print(f"Tigerbot: {response}")