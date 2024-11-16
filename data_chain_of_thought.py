from openai import OpenAI
import json
import settings
# OpenAI API 키 설정
# OpenAI API 키 설정
client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
)

# 청크 파일 경로
chunked_file_path = r"C:\Users\82104\Desktop\AI반도체경진대회\정책자료\정책계획자료\chunks\chunked_file_1.json"

def process_policy_document_with_openai(document_text):
    # Step 1: 특정 대상에 대한 정보 추출
    extract_messages = [
            {"role": "system", "content": "당신은 개인사업자, 자영업자, 소상공인 관련 정책 전문가입니다."},
            {"role": "user", "content": f"""
    아래는 연금개혁 추진계획입니다. 이 내용을 바탕으로 개인사업자, 자영업자, 소상공인을 대상으로 하는 
    정책 정보를 추출해주세요. 그들에게 특히 중요한 정책 내용(예: 보험료 지원, 연금 구조 변경 등)을 중심으로 상세히 설명해주세요.

    내용:
    {document_text}
    """}
        ]
    
    extract_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=extract_messages,
            max_tokens=1200,
            temperature=0.5
        )
    extracted_info = extract_response.choices[0].message.content

    # Step 2: 구체적인 심층 분석
    analyze_messages = [
            {"role": "system", "content": "당신은 개인사업자, 자영업자, 소상공인 관련 정책 전문가입니다."},
            {"role": "user", "content": f"""
    다음은 개인사업자, 자영업자, 소상공인과 관련된 정책 정보입니다.
    이 내용을 더 구체적으로 분석하고 아래 항목에 따라 정리해주세요:
    1. 대상자 조건이 어떻게 되는지
    2. 대상자에게 어떤 혜택이 주어지는지
    3. 정책 시행의 배경과 예상 효과.
    4. 정책에 따라 필요한 행동 계획.

    정책 정보:
    {extracted_info}
    """}
        ]
    analyze_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=analyze_messages,
            max_tokens=1200,
            temperature=0.5
        )
    analyzed_info = analyze_response.choices[0].message.content

    # Step 3: 구조화된 정보로 재구성
    structure_message = [
            {"role": "system", "content": "당신은 개인사업자, 자영업자, 소상공인 관련 정책 전문가입니다."},
            {"role": "user", "content": f"""
    다음은 분석된 정책 정보입니다. 이를 아래 항목으로 구조화하여 정리해주세요:
    - 정책 이름:
    - 제공되는 혜택:
    - 적용 대상:
    - 세부 내용:
    - 기대 효과:
    - 시행 시기 및 절차:
    - 추가적으로 알아야 할 점:

    분석 내용:
    {analyzed_info}
    """}
        ]
    structure_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=structure_message,
            max_tokens=1200,
            temperature=0.5
        )
    structured_info = structure_response.choices[0].message.content
    '''
    # Step 5: 사용자 맞춤형 재가공
    refine_message = [
            {"role": "system", "content": "당신은 개인사업자, 자영업자, 소상공인 관련 정책 전문가입니다."},
            {"role": "user", "content": f"""
    다음은 구조화된 정책 정보입니다. 이를 읽기 쉬운 형태로 재가공하세요.
    대상 독자가 정책을 쉽게 이해할 수 있도록, 전문 용어를 최소화하고 친근한 어조로 작성해주세요.
    또한, 각각의 정책에 대해 적용 사례를 포함하여 설명해주세요.

    구조화된 정보:
    {structured_info}
    """}
        ]
    refine_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=refine_message,
            max_tokens=1200,
            temperature=0.5
        )
    refined_text = refine_response.choices[0].message.content

    return refined_text
    '''
    return structured_info
# 각 청크를 처리하는 루프
with open(chunked_file_path, "r", encoding="utf-8") as file:
    chunks = json.load(file)

results = []
for chunk in chunks:
    result = process_policy_document_with_openai(chunk)
    results.append(result)

# 처리 결과를 새로운 JSON 파일로 저장
output_path = r"C:\Users\82104\Desktop\AI반도체경진대회\prompt_evaluation\policy_news"
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(results, output_file, ensure_ascii=False, indent=4)

print(f"처리된 결과가 {output_path}에 저장되었습니다.")
