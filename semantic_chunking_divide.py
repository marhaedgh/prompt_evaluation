import json
import os
from transformers import AutoTokenizer, AutoModel
import torch
from llama_index.embeddings.base import BaseEmbedding
from llama_index.node_parser import SemanticSplitterNodeParser
from llama_index import Document
from math import ceil

# 'sentence-transformers/all-MiniLM-L6-v2' 모델 사용
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # 마지막 히든 스테이트
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=4000  # 최대 토큰 길이 설정
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return mean_pooling(outputs, inputs['attention_mask']).squeeze().numpy()

# BaseEmbedding을 상속받는 커스텀 클래스 정의
class CustomEmbedding(BaseEmbedding):
    def get_text_embedding(self, text: str):
        return get_embedding(text)

    def _get_text_embedding(self, text: str):
        return get_embedding(text)

    def _get_query_embedding(self, query: str):
        return get_embedding(query)

    async def _aget_query_embedding(self, query: str):
        return get_embedding(query)

# CustomEmbedding 인스턴스 생성
custom_embed_model = CustomEmbedding()

# SemanticSplitterNodeParser 설정
splitter = SemanticSplitterNodeParser(
    buffer_size=10,  # 청크 크기를 늘리기 위해 buffer_size 증가
    breakpoint_percentile_threshold=95,
    embed_model=custom_embed_model
)

# 큰 JSON 파일을 읽고 텍스트 추출 및 청크로 분할
def process_large_json(file_path, output_dir, text_key, chunks_per_file=1000):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # JSON 파일에서 텍스트 추출 (딕셔너리 형태)
    raw_texts = [item.get(text_key, "") for item in data.values()]
    print(f"Extracted {len(raw_texts)} texts from JSON.")

    # Document 객체 생성
    documents = [Document(text=text) for text in raw_texts if text.strip()]
    print(f"Created {len(documents)} Document objects.")

    all_chunks = []
    for idx, doc in enumerate(documents):
        nodes = splitter.get_nodes_from_documents([doc])
        chunks = [node.text for node in nodes]
        all_chunks.extend(chunks)
        print(f"Document {idx + 1}: {len(chunks)} chunks created.")

    if not all_chunks:
        print("No chunks generated. Check the input data or splitter logic.")
        return

    print(f"Total chunks: {len(all_chunks)}")
    save_chunks_to_multiple_files(all_chunks, output_dir, chunks_per_file)

# 분할된 청크를 여러 JSON 파일로 저장
def save_chunks_to_multiple_files(chunks, output_dir, chunks_per_file=1000):
    os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 없으면 생성
    total_chunks = len(chunks)
    num_files = ceil(total_chunks / chunks_per_file)
    
    for i in range(num_files):
        start = i * chunks_per_file
        end = start + chunks_per_file
        file_chunks = chunks[start:end]
        
        output_file = os.path.join(output_dir, f"chunked_file_{i+1}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(file_chunks, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(file_chunks)} chunks to {output_file}")

# 실행
input_file = ""  # JSON 파일 경로 
output_directory = ""  # 청크 결과 저장 디렉토리
process_large_json(input_file, output_directory, text_key="text", chunks_per_file=1000)
