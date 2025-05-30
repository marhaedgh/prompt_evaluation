import settings
from openai import OpenAI
from typing import List, Dict
import numpy as np

class GPTScoreEvaluator:
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

    def gpt_logprob_score(self, text: str, reference: str, task_desc: str, aspect_desc: str, model="gpt-3.5-turbo") -> float:
        """
        로그 확률 기반의 GPTScore를 계산하는 함수.
        """
        # 평가 프롬프트 생성
        prompt = (
            f"{task_desc}\n"
            f"평가 기준: {aspect_desc}\n"
            f"원문 텍스트: {reference}\n"
            f"요약 텍스트: {text}\n"
            "위 요약의 품질을 평가하기 위한 로그 확률을 반환합니다."
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

    def normalize_score(self, logprob: float, min_logprob: float=-9, max_logprob: float=0) -> float:
        """
        로그 확률 점수를 0~100 점 범위로 정규화하는 함수.
        """
        # 점수를 0에서 100 범위로 스케일링
        normalized_score = (logprob - min_logprob) / (max_logprob - min_logprob) * 100
        # 0과 100 범위로 클리핑하여 점수가 범위를 초과하지 않도록 함
        normalized_score = max(0, min(100, normalized_score))
        return normalized_score

    def evaluate_summary_logprob(self, text: str, reference: str) -> Dict[str, float]:
        """
        여러 기준에 대해 로그 확률 기반으로 요약의 품질을 평가하는 함수.
        """
        task_desc = "다음 텍스트를 요약하고 자영업자를 위한 정책 정보 제공에 맞게 조정하세요."

        criteria = {
            "Relevance": "요약이 원문과 얼마나 관련이 있는지 확인",
            "Informativeness": "요약이 원문에서 중요한 아이디어를 얼마나 잘 포착했는지 확인",
            "Understandability": "요약이 자영업자가 이해하기 쉽게 작성되었는지 확인",
            "Specificity": "요약이 구체적인 정보를 제공하고 있는지 확인",
            "Engagement": "요약이 독자의 관심을 끌고 자영업자의 행동을 유도할 수 있는지 확인"
        }
        scores = {}
        for aspect, desc in criteria.items():
            score = self.gpt_logprob_score(text, reference, task_desc, desc)
            scores[aspect] = score
        # 각 점수에 대해 정규화 적용
        normalized_scores = {aspect: self.normalize_score(score) for aspect, score in scores.items()}
        return normalized_scores

# 사용 예제
api_key = settings.OPENAI_API_KEY
client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
)
print(client.api_key)
evaluator = GPTScoreEvaluator(api_key=client.api_key)

reference_text = "아이엘사이언스의 자회사 아이트로닉스는 차량용 복합기능형 졸음 방지 단말기 특허를 출원했다고 4일 밝혔다..."
summary_text = "아이엘사이언스의 자회사 아이트로닉스가 차량용 졸음 방지 단말기 특허를 출원했습니다..."

# 평가 수행
GPTscore = evaluator.evaluate_summary_logprob(summary_text, reference_text)
scores = {}
for i,j in GPTscore.items():
    scores[i] = j
print(scores)
print("GPTScore:\n" + "\n".join([f"  {aspect}: {score:.2f}" for aspect, score in GPTscore.items()]))
