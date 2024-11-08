import settings 
from openai import OpenAI
from bert_score import score
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict
import numpy as np

# OpenAI API 키 설정
#openai.api_key = 'API 입력ㅎㅇㅇ'
client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
)
# 1. 원자적 사실 추출 함수 (OpenAI API 사용)
def extract_atomic_facts(text):
    messages = [
        {"role": "system", "content": "당신은 글에서 최소 단위 정보들 추출의 전문가입니다. "},
        {"role": "user", "content": f"다음 텍스트에서 각 문장의 핵심 정보를 담은 최소 단위 정보를 추출하세요 : \n\n{text}\n\n각 사실을 한 줄씩 나열하세요"
    }
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=150,
        temperature=0
    )
    facts = response.choices[0].message.content.strip().split('\n')
    return [fact.strip() for fact in facts if fact.strip()]

# 2. 원자적 사실 간 비교 함수 (BERTScore 사용)
def compare_atomic_facts(ref_facts, gen_facts):
    matched_facts = []
    for gen_fact in gen_facts:
        P, R, F1 = score([gen_fact] * len(ref_facts), ref_facts, lang="ko")
        max_score_idx = F1.argmax()
        max_score = F1[max_score_idx].item()

        if max_score >= 0.8:  # 일정 유사도 이상일 때 일치로 간주
            matched_facts.append((gen_fact, ref_facts[max_score_idx]))
    return matched_facts

# 3. 논리적 흐름 확인 함수
def verify_logical_chain(matched_facts, ref_facts):
    for i in range(1, len(matched_facts)):
        prev_fact = matched_facts[i-1][1]  # 이전에 매칭된 원자적 사실
        curr_fact = matched_facts[i][1]    # 현재 매칭된 원자적 사실
        # 현재 사실이 이전 사실의 순서 이후에 나타나는지 확인
        if curr_fact not in ref_facts[ref_facts.index(prev_fact) + 1:]:
            print("논리적 흐름에 문제가 있습니다.")
            return False
    return True

# 4. 사실적 일관성 점수 계산 함수
def factual_consistency_score(ref_facts, gen_facts):
    matched_facts = compare_atomic_facts(ref_facts, gen_facts)
    logical_consistency = verify_logical_chain(matched_facts, ref_facts)
    
    accuracy_score = len(matched_facts) / len(ref_facts) * 100
    coherence_penalty = 10 if not logical_consistency else 0  # 일관성 부족시 감점
    
    #final_score = max(0, accuracy_score - coherence_penalty) 둘이 합친건데 일단 따로 하는게 나은듯
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
            if similarity > 0.85:  # 유사도가 0.85 이상일 경우 중복으로 간주
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
        return scores

# 8. 자동 평가 함수
def evaluate_model(dataset, api_key):
    index = 0
    while index < len(dataset):
        item = dataset[index]
        document = item['document']
        ref_summary = item['summary']
        
        print("\nOriginal Document:\n", document)
        print("\nReference Summary:\n", ref_summary)

        custom_summary = input("\nEnter your custom summary:\n")

        ref_facts = extract_atomic_facts(ref_summary)
        gen_facts = extract_atomic_facts(custom_summary)
        src_facts = extract_atomic_facts(document)
        gen_match_acc, gen_logical_consist = factual_consistency_score(src_facts, gen_facts)
        ref_match_acc, ref_logical_consist = factual_consistency_score(src_facts, ref_facts)
        bertscore = calculate_bertscore(ref_summary, custom_summary) 
        gen_concise = evaluate_redundancy_with_embeddings(custom_summary)
        ref_concise = evaluate_redundancy_with_embeddings(ref_summary)
        #추가된 GPTScore
        evaluator = GPTScoreEvaluator(api_key=api_key)
        evaluation_scores = evaluator.evaluate_summary_logprob(custom_summary, document)
        # 최종 평가 결과 출력 
        print(f"\n평가 결과:")
        print(f"데이터 요약vs모델 요약 BERTScore: {bertscore:.2f}")
        print(f"사실 매칭 점수: model-{gen_match_acc:.2f} || ref-{ref_match_acc:.2f}")
        print(f"논리적 일관성 점수: model-{gen_logical_consist:.2f} || ref-{ref_logical_consist:.2f}")
        print(f"간결성 점수: model-{gen_concise:.2f} || ref-{ref_concise:.2f}")
        print("평가 점수 (로그 확률 기반):", evaluation_scores)
        
        continue_evaluation = input("\nDo you want to evaluate another summary? (Enter 'y' for yes, any other key to stop): ")
        if continue_evaluation.lower() != 'y':
            print("Evaluation session ended.")
            break
        index += 1

# 9. 데이터셋 로드 및 평가 실행
dataset = load_dataset("daekeun-ml/naver-news-summarization-ko", cache_dir='C:\\Users\\82104\\Desktop\\RAG_테크닉\\허깅페이스', split="test")
evaluate_model(dataset, client.api_key)