import settings 
from openai import OpenAI
from bert_score import score
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime
import os
import pandas as pd

import json


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
            model="gpt-3.5-turbo",
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
def evaluate_model(dataset, client: OpenAI):
    index = 0
    scores_storage = []
    while index < 2:#len(dataset):  # 데이터셋 크기만큼 반복
        print(index)
        item = dataset[index]
        document = ''
        for page, content in item.items():
            if 'text' in content:  # 'text' 키가 있는 경우만 처리
                document+=content['text']  # 'document' 키로 변환
            if 'table' in content:
                document+=content['table']
        evaluator = FactualConsistencyEvaluator(client)
        messages = [
            {"role": "system", "content": 
"""
*역할*
당신은 복잡한 정보를 접근 가능하고 공감 가는 요약으로 변환하는 전문가입니다. 개인사업자를 위한 내용을 명확하고 친근한 언어로 재구성해 전달하는 것이 목표입니다. 주어진 텍스트를 체계적 사고 과정(Chain of Thought)을 통해 1000자 이내로 요약하고, 독자들에게 필요한 핵심 정보를 전달하는 것이 목적입니다.

*맥락*
목표: 텍스트를 깊이 이해하고, 개인사업자들이 관심을 가질 만한 주요 포인트를 찾아 전달하는 것
독자: 긴 글을 간단하게 이해하고 싶어하는 '개인사업자'. 복잡한 용어를 지양하고, 쉽게 이해할 수 있는 친근한 표현을 사용
접근 방식: 구조가 명확하고, 독자가 공감할 수 있는 어조 사용

*절차*
1. 텍스트 이해: 내용을 충분히 검토해 주요 아이디어, 목표, 뉘앙스를 파악합니다.
2. 주제와 목적 확인: 텍스트의 중심 주제와 개인사업자에게 유익한 요소를 식별합니다.
3. 키워드 추출: 내용을 요약하는 데 중요한 핵심 키워드와 개념을 선별합니다.
4. 내용 구조화: 정보를 논리적으로 재구성하고, 단락 사이의 연결이 자연스럽도록 배치합니다.
5. 단락 및 소제목 구성: 명확한 구조를 위해 짧은 단락과 흥미로운 소제목을 작성합니다.
6. 어조 및 스타일 설정: 친근하고 공감 가는 어조를 유지하며, 가능한 한 쉬운 표현을 사용합니다.
7. 요약 작성: 독자의 흥미를 끌 수 있는 요약을 작성합니다.
8. 독자의 흥미 유발: 첫 문장을 인상 깊게 작성하여 독자의 관심을 끌고, 개인사업자에게 유용한 내용을 강조합니다.
9. 검토 및 다듬기: 요약이 간결하고 명확하며 일관성이 있는지 검토합니다.
10. 글자 수 및 구조 최종 확인: 요약이 1000자 이내이며 논리적 흐름을 유지하도록 조정합니다.

*지침*
- 편안하고 친근한 어조
- 공감 가는 표현 사용
- 이해하기 쉬운 용어
- 독자의 관심을 끌 만한 인상적인 첫 단락
- 새로운 용어는 간단하게 설명
- 1000자 이내의 글자 수 및 짧은 단락 유지"""},
            {"role": "user", "content": f"""
             *요청*
1. 문서 요약을 수행할 때 독자(개인사업자)가 가장 관심을 가질 만한 포인트에 집중하세요. 
2. 내용을 명확하게 전달하되, 독자의 흥미를 유발하기 위해 일상에서의 비유나 예시를 포함해 주세요.
3. 중요한 정보가 구체적으로 이해될 수 있도록 관련 배경 지식을 간단히 덧붙여 주세요.
4. 독자가 글의 내용을 통해 어떤 행동을 해야하는지 구체적으로 가이드라인을 제시해주세요.\n
요약 텍스트: {document}
"""
             }
        ]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1200,
            temperature=0.5
        )
        ref_summary = response.choices[0].message.content
        # 평가 수행
        ref_accuracy, ref_logical_consistency = evaluator.factual_consistency_score(document, ref_summary)
        ref_accuracy = round(ref_accuracy, 2)
        ref_logical_consistency = round(ref_logical_consistency,2)
        ref_concise = round(evaluate_redundancy_with_embeddings(ref_summary), 2)
        bertscore = round(calculate_bertscore(document, ref_summary), 2)*100
        evaluator = GPTScoreEvaluator(client.api_key)

        GPTscore = evaluator.evaluate_summary_logprob(ref_summary, document)
        scores={}
        scores["accuracy"] = round(ref_accuracy,2)
        scores["consistency"] = round(ref_logical_consistency,2)
        scores["conciseness"] = round(ref_concise,2)
        scores["bertscore"] = round(bertscore,2)
        for i,j in GPTscore.items():
            scores[i] = j
        scores_storage.append(scores)
        index += 1
    
    # 오늘 날짜로 폴더 생성
    date_str = datetime.now().strftime("%Y-%m-%d")
    directory = f"evaluation_scores/{date_str}"
    os.makedirs(directory, exist_ok=True)
    # 파일 번호 확인 및 넘버링
    file_count = len([f for f in os.listdir(directory) if f.startswith("policy_summary_GPT_scores_")]) + 1
    file_name = f"policy_summary_GPT_scores_{file_count}.csv"
    file_path = os.path.join(directory, file_name)
    df = pd.DataFrame(scores_storage)
    # 파일 저장
    df.to_csv(file_path, index=False)
    print("CSV 파일에 모든 평가 점수를 저장했습니다.")


# JSON 파일 데이터셋 로드 함수
def load_json_dataset_from_folder(folder_path):
    dataset = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):  # JSON 파일만 처리
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)
                # \x00 문자 제거
                for page, content in data.items():
                    if 'text' in content:  # 'text' 키가 있는지 확인
                        cleaned_text = content['text'].replace("\x00", "")  # NULL 문자 제거
                        content['text'] = cleaned_text  # 클린된 텍스트로 업데이트
                dataset.append(data)
    return dataset

# JSON 데이터셋 로드 및 평가 실행
folder_path = r"C:\Users\82104\Desktop\AI반도체경진대회\정책자료\json"  # JSON 파일이 저장된 폴더 경로
dataset = load_json_dataset_from_folder(folder_path)
evaluate_model(dataset, client)

