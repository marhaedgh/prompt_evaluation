# 요약 모델 평가 시스템 (Summary Model Evaluation System) eval.py
OpenAI API, BERTScore, Sentence-BERT 등을 활용하여 모델이 생성한 요약이 원문과 얼마나 잘 일치하는지, 간결성 및 논리적 일관성이 있는지 등을 측정합니다.  
BERTScore, GPTScore, SummEval 등의 논문을 참고했습니다.  
gptscore_logprob은 GPTScore 평가 방식만 따로 빼둔 것입니다.  
## 기능 (Features)

1. **atomic facts 추출**: GPT를 활용하여 텍스트에서 최소 단위 정보(atomic facts)를 추출합니다. 
2. **사실 매칭 및 비교**: 1의 기능으로 생성된 요약과 참조 요약의 각 atomic facts을 BERTScore를 통해 비교하여 일치도를 계산합니다.
3. **논리적 흐름 검사**: 요약 문서의 원자적 사실의 순서를 분석하여 논리적 일관성이 유지되고 있는지 확인합니다.
4. **간결성 평가**: Sentence-BERT 기반의 문장 간 유사도 분석을 통해 중복도를 평가하고 간결성을 측정합니다.
5. **GPTScore 평가**: GPT의 로그 확률 기반으로 요약의 관련성, 정보성, 이해 용이성, 구체성 등을 점수화합니다.
6. **BERTScore**: 참조 요약과 모델 요약 간의 유사성을 측정합니다.
7. **자동화된 요약 평가**: 데이터셋을 기반으로 요약을 평가하고, 결과를 사용자가 직접 확인할 수 있는 인터페이스를 제공합니다.

## 설치 (Installation)

### 사전 요구사항

- Python 3.8 이상
- `OpenAI API` 키가 필요합니다. 
### 필요한 패키지 설치


###개발환경
requirements.txt 다운받아서 사용
pip install -r requirements.txt

##사용법
API 키 설정: 따로 settings.py 파일을 만들어 OpenAI API 키를 추가합니다.  
데이터셋 준비: Hugging Face에서 데이터셋을 로드하고 평가할 데이터셋을 설정합니다. cache_dir 매개변수에 데이터를 저장할 위치를 넣습니다. 개인 로컬이나 클라우드 어디든.??
평가 함수 실행: evaluate_model 함수는 사용자가 직접 작성한 요약문(custom_summary)을 네이버 기사 원문(document) 및 네이버 제공 요약문(ref_summary)과 비교하여 평가를 진행합니다.   
이 함수는 while 루프를 통해 각 문서에 대해 반복적으로 평가를 수행합니다.  
함수를 실행하면, 모델로 요약한 문서를 input할 수 있습니다.  
이거 자동화는 모델 서빙에서 받는게 익숙해지면 빨리 만들겠습니다ㅏㅏ

