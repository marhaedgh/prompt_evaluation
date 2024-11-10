import settings 
from openai import OpenAI
from bert_score import score
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# OpenAI API 키 설정
client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
)

class FactualConsistencyEvaluator:
    def __init__(self, client: OpenAI):
        self.client = client
    # 1. 원자적 사실 추출 함수 (OpenAI API 사용)
    def extract_atomic_facts(self, text: str) -> List[str]:
        messages = [
            {"role": "system", "content": "당신은 글에서 최소 단위 정보들 추출의 전문가입니다."},
            {"role": "user", "content": f"다음 텍스트에서 각 문장의 핵심 정보를 담은 최소 단위 정보를 추출하세요:\n\n{text}\n\n각 사실을 한 줄씩 나열하세요"}
        ]
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=150,
            temperature=0
        )
        facts = response.choices[0].message.content.strip().split('\n')
        return [fact.strip() for fact in facts if fact.strip()]
    # 2. 원자적 사실 간 비교 함수 (BERTScore 사용)
    def compare_atomic_facts(self, ref_facts: List[str], gen_facts: List[str]) -> List[Tuple[str, str]]:
        matched_facts = []
        for gen_fact in gen_facts:
            P, R, F1 = score([gen_fact] * len(ref_facts), ref_facts, lang="ko")
            max_score_idx = F1.argmax()
            max_score = F1[max_score_idx].item()

            if max_score >= 0.8:  # 일정 유사도 이상일 때 일치로 간주
                matched_facts.append((gen_fact, ref_facts[max_score_idx]))
        return matched_facts
    
    # 3. 논리적 흐름 확인 함수
    def verify_logical_chain(self, matched_facts: List[Tuple[str, str]], ref_facts: List[str]) -> float:
        """
        Parameters:
        - matched_facts: 생성된 사실과 참조 사실의 매칭된 쌍 리스트
        - ref_facts: 참조 사실들의 리스트
        Returns:
        - 논리적 흐름 점수 (0에서 100 사이의 값)
        """
        if not matched_facts:
            return 0.0  # 매칭된 사실이 없으면 점수는 0
        total_pairs = len(matched_facts) - 1
        if total_pairs <= 0:
            return 100.0  # 매칭된 사실이 하나뿐이면 논리적 흐름에 문제가 없다고 간주
        consistent_pairs = 0
        for i in range(1, len(matched_facts)):
            prev_fact = matched_facts[i - 1][1]  # 이전에 매칭된 참조 사실
            curr_fact = matched_facts[i][1]      # 현재 매칭된 참조 사실

            # 이전 사실의 다음 위치부터 현재 사실이 나타나는지 확인
            if curr_fact in ref_facts[ref_facts.index(prev_fact) + 1:]:
                consistent_pairs += 1

        # 논리적 흐름 점수 계산
        logical_flow_score = (consistent_pairs / total_pairs) * 100
        return logical_flow_score
    
    #4. 위 FactualConsistencyEvaluator 종합하여 점수 반환
    def factual_consistency_score(self, ref_text: str, gen_text: str) -> Tuple[float, bool]:
        ref_facts = self.extract_atomic_facts(ref_text)
        gen_facts = self.extract_atomic_facts(gen_text)
        matched_facts = self.compare_atomic_facts(ref_facts, gen_facts)
        logical_consistency = self.verify_logical_chain(matched_facts, ref_facts)
        
        accuracy_score = len(matched_facts) / len(ref_facts) * 100 if ref_facts else 0
        return accuracy_score, logical_consistency

# 5. BERTScore 계산 함수 (참조 요약과 모델 요약 비교)
def calculate_bertscore(ref_summaries, model_summaries, lang='ko'):
    if isinstance(ref_summaries, str):
        ref_summaries = [ref_summaries]
    if isinstance(model_summaries, str):
        model_summaries = [model_summaries]
    P, R, F1 = score(model_summaries, ref_summaries, lang=lang)
    return F1.mean().item()

#6. 간결성(문장 중복 정도)
def evaluate_redundancy_with_embeddings(summary_text):
    # Sentence-BERT 모델 로드
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 요약문을 문장 단위로 분할
    sentences = summary_text.split('.')
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # 코사인 유사도를 기반으로 유사성 측정
    redundancy_penalty = 0
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
            if similarity > 0.75:  # 유사도가 0.85 이상일 경우 중복으로 간주
                redundancy_penalty += 10  # 중복 문장당 10점 감점
    # 간결성 점수 계산
    conciseness_score = max(0, 100 - redundancy_penalty)
    return conciseness_score

 
# 7. GPTScore 구현

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

# 8. 자동 평가 함수
def evaluate_model(dataset, client):
    index = 0
    scores_storage = []
    while index < len(dataset):
        item = dataset[index]
        document = item['document']
        ref_summary = item['summary']
        print(index)
        evaluator = FactualConsistencyEvaluator(client)
        # 평가 수행
        ref_accuracy, ref_logical_consistency = evaluator.factual_consistency_score(document, ref_summary)
        ref_concise = evaluate_redundancy_with_embeddings(ref_summary)
        bertscore = calculate_bertscore(document, ref_summary)
        #추가된 GPTScore
        evaluator = GPTScoreEvaluator(client)
        GPTscore = evaluator.evaluate_summary_logprob(ref_summary, document)

        scores={}
        scores["accuracy"] = ref_accuracy
        scores["consistency"] = ref_logical_consistency
        scores["conciseness"] = ref_concise
        scores["bertscore"] = bertscore
        for i,j in GPTscore.items():
            scores[i] = j
        print(scores)
        scores_storage.append(scores)
        continue_evaluation = input("\nDo you want to evaluate another summary? (Enter 'y' for yes, any other key to stop): ")
        if continue_evaluation.lower() != 'y':
            print("Evaluation session ended.")
            break
        index += 1

# 9. 데이터셋 로드 및 평가 실행
dataset = load_dataset("daekeun-ml/naver-news-summarization-ko", cache_dir='C:\\Users\\82104\\Desktop\\RAG_테크닉\\허깅페이스', split="test")
evaluate_model(dataset, client)