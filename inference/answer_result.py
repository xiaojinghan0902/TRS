from typing import Optional, List

class AnswerResult:     # 流式消息实体
    history: List[List[str]] = []
    llm_output: Optional[dict] = None