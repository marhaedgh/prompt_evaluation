import settings
from openai import OpenAI
from typing import List, Dict
import numpy as np
from datasets import load_dataset

class GPTScoreTitleEvaluator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    def get_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        max_tokens=500,
        temperature=0,
        stop=None,
        seed=123,
        tools=None,
        logprobs=True,
        top_logprobs=5
    ) -> dict:
        """
        OpenAI API 호출을 통해 모델의 응답을 가져오는 함수.
        """
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop,
            "seed": seed,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
        }
        if tools:
            params["tools"] = tools

        completion = self.client.chat.completions.create(**params)
        return completion

    def title_score(self, title: str, reference: str, task_desc: str, aspect_desc: str, model="gpt-3.5-turbo") -> float:
        """
        로그 확률 기반의 title의 점수를 계산하는 함수.
        """
        # 평가 프롬프트 생성
        prompt = (
            f"{task_desc}\n"
            f"평가 기준: {aspect_desc}\n"
            f"원문 텍스트: {reference}\n"
            f"타이틀: {title}\n"
            "위 타이틀의 품질을 평가하기 위한 로그 확률을 반환합니다."
        )

        # OpenAI API 호출을 통한 응답과 로그 확률 수집
        response = self.get_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            logprobs=True
        )

        # 로그 확률 추출
        logprobs_values = [
            logprob.logprob for logprob in response.choices[0].logprobs.content[0].top_logprobs
        ]

        # 평균 로그 확률 계산
        average_logprob = np.mean(logprobs_values)
        return average_logprob

    def evaluate_title_logprob(self, title: str, reference: str) -> Dict[str, float]:
        """
        여러 기준에 대해 로그 확률 기반으로 타이틀의 품질을 평가하는 함수.
        """
        task_desc = "다음 텍스트에 대해 타이틀의 매력과 일치도를 평가하세요."
        
        criteria = {
            "Engagement": "타이틀이 독자의 관심을 끌고 호기심을 유발하는지 확인",
            "Content Alignment": "타이틀이 원문과 얼마나 내용적으로 일치하는지 확인"
        }
        
        scores = {}
        for aspect, desc in criteria.items():
            score = self.title_score(title, reference, task_desc, desc)
            scores[aspect] = score
        return scores


api_key = settings.OPENAI_API_KEY
evaluator = GPTScoreTitleEvaluator(api_key=api_key)


dataset = load_dataset("daekeun-ml/naver-news-summarization-ko", cache_dir='C:\\Users\\82104\\Desktop\\RAG_테크닉\\허깅페이스', split="test")

# 평가 수행
index = 0
while index < len(dataset):
    item = dataset[index]
    document = item['document']
    title_text = item['title']
    
    print("\nReference title:\n", title_text)

    custom_title = input("\nEnter your custom title:\n")
    evaluation_scores = evaluator.evaluate_title_logprob(document, custom_title)
    
    print("타이틀 평가 점수 (로그 확률 기반):", evaluation_scores)
    continue_evaluation = input("\nDo you want to evaluate another summary? (Enter 'y' for yes, any other key to stop): ")
    if continue_evaluation.lower() != 'y':
        print("Evaluation session ended.")
        break
    index += 1