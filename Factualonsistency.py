import settings 
from openai import OpenAI
from bert_score import score
from datasets import load_dataset
from typing import List, Tuple

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
        print(facts)
        return [fact.strip() for fact in facts if fact.strip()]
    # 2. 원자적 사실 간 비교 함수 (BERTScore 사용)
    def compare_atomic_facts(self, ref_facts: List[str], gen_facts: List[str]) -> List[Tuple[str, str]]:
        matched_facts = []
        for gen_fact in gen_facts:
            P, R, F1 = score([gen_fact] * len(ref_facts), ref_facts, lang="ko")
            print(P,R,F1)
            max_score_idx = F1.argmax()
            max_score = F1[max_score_idx].item()

            if max_score >= 0.8:  # 일정 유사도 이상일 때 일치로 간주
                matched_facts.append((gen_fact, ref_facts[max_score_idx]))
        print(matched_facts)
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

    def factual_consistency_score(self, ref_text: str, gen_text: str) -> Tuple[float, bool]:
        ref_facts = self.extract_atomic_facts(ref_text)
        gen_facts = self.extract_atomic_facts(gen_text)
        matched_facts = self.compare_atomic_facts(ref_facts, gen_facts)
        logical_consistency = self.verify_logical_chain(matched_facts, ref_facts)
        
        accuracy_score = len(matched_facts) / len(ref_facts) * 100 if ref_facts else 0
        return accuracy_score, logical_consistency

# 4. 사실적 일관성 점수 계산 함수
def evaluate_model(dataset, client):
    print(client, type(client))
    evaluator = FactualConsistencyEvaluator(client)
    
    for item in dataset:
        document = item['document']
        ref_summary = item['summary']
        
        print("\nOriginal Document:\n", document)
        print("\nReference Summary:\n", ref_summary)

        custom_summary = input("\nEnter your custom summary:\n")

        # 평가 수행
        gen_accuracy, gen_logical_consistency = evaluator.factual_consistency_score(document, custom_summary)
        ref_accuracy, ref_logical_consistency = evaluator.factual_consistency_score(document, ref_summary)

        # 결과 출력
        print("\n평가 결과:")
        print(f"사용자 요약 - 정확도: {gen_accuracy:.2f}%, 논리적 일관성: {gen_logical_consistency:.2f}")
        print(f"참고 요약 - 정확도: {ref_accuracy:.2f}%, 논리적 일관성: {ref_logical_consistency:.2f}")
        
        if input("\nEvaluate another summary? (y/n): ").lower() != 'y':
            print("Evaluation session ended.")
            break
        

# 9. 데이터셋 로드 및 평가 실행
dataset = load_dataset("daekeun-ml/naver-news-summarization-ko", cache_dir='C:\\Users\\82104\\Desktop', split="test")
evaluate_model(dataset, client)