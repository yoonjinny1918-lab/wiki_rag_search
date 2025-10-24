import os
import pandas as pd
import streamlit as st
from openai import OpenAI
from elasticsearch import Elasticsearch

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="정케빈의 AI 위키 검색기",
    page_icon="📘",
    layout="wide"
)

# Streamlit 스타일 커스터마이징
st.markdown("""
    <style>
    body {
        background-color: #f8fafc;
    }
    .main-title {
        text-align: center;
        color: #1e3a8a;
        font-size: 2.5rem;
        font-weight: bold;
        margin-top: -20px;
    }
    .sub-title {
        text-align: center;
        color: #475569;
        font-size: 1.1rem;
    }
    .result-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }
    .wiki-card {
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 10px;
        font-size: 0.9rem;
    }
    .footer {
        color: #94a3b8;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# OpenAI & Elasticsearch 연결
# -----------------------------
client = OpenAI(api_key=st.secrets["api_key"])
ELASTIC_CLOUD_ID = st.secrets["elastic_cloud_key"]
ELASTIC_API_KEY = st.secrets["elastic_api_key"]

es = Elasticsearch(
    cloud_id=ELASTIC_CLOUD_ID,
    api_key=ELASTIC_API_KEY
)

# 연결 테스트
try:
    es.info()
except Exception as e:
    st.error(f"❌ Elasticsearch 연결 실패: {e}")
    st.stop()

# -----------------------------
# 헤더 및 소개 섹션
# -----------------------------
st.markdown("<h1 class='main-title'>📘 한글로 답변하는 위키 기반 AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Semantic Search + RAG 기반 | Powered by Elasticsearch & OpenAI</p>", unsafe_allow_html=True)
st.divider()

with st.expander("📄 서비스 설명", expanded=False):
    st.caption("""
    이 서비스는 **영문 위키피디아 데이터셋(25,000건)**을 기반으로
    한국어 질문에 대해 **의미 검색(Semantic Search)** 및 **RAG(Retrieval-Augmented Generation)** 기술을 활용해
    한글로 답변을 제공합니다.

    **예시 질문**
    - 대서양은 몇 번째로 큰 바다인가?
    - 대한민국의 수도는?
    - 도요타에서 가장 많이 팔리는 차는?

    **데이터 출처**
    - [Wikipedia Embeddings Dataset](https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip)
    - [데이터 설명](https://weaviate.io/developers/weaviate/tutorials/wikipedia)
    """)

# -----------------------------
# 질문 입력 섹션
# -----------------------------
st.markdown("### 💬 질문을 입력하세요")
question = st.text_input("Prompt", placeholder="예: 대서양은 몇 번째로 큰 바다인가?")
submit = st.button("🚀 질문하기")

# -----------------------------
# 처리 로직
# -----------------------------
if submit and question:
    with st.spinner("🤖 Kevin AI가 답변을 생성 중입니다..."):
        try:
            # Step 1. 한국어 → 영어 번역
            translation = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Translate the following Korean question into English: {question}"}]
            ).choices[0].message.content.strip()

            # Step 2. 질문 임베딩 생성
            embedding = client.embeddings.create(
                input=[translation],
                model="text-embedding-ada-002"
            ).data[0].embedding

            # Step 3. Elasticsearch 검색
            response = es.search(
                index="wikipedia_vector_index",
                knn={
                    "field": "content_vector",
                    "query_vector": embedding,
                    "k": 5,
                    "num_candidates": 50
                }
            )

            # Step 4. 상위 문서 요약 및 답변
            top_hit = response['hits']['hits'][0]['_source']
            summary = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant that answers in Korean based on the given context."},
                    {"role": "user", "content": f"질문: {question}\n\n참고 문서: {top_hit['text']}"}
                ]
            )

            # -----------------------------
            # 결과 출력
            # -----------------------------
            st.divider()
            st.markdown("### 🧠 AI의 답변")
            st.markdown(f"<div class='result-card'>{summary.choices[0].message.content}</div>", unsafe_allow_html=True)

            st.markdown("### 🔎 검색된 문서 목록")
            for hit in response['hits']['hits']:
                title = hit['_source']['title']
                url = hit['_source']['url']
                score = round(hit['_score'], 2)
                st.markdown(f"<div class='wiki-card'>🔗 [{title}]({url})<br/>점수: {score}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ 오류 발생: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("<div class='footer'>© 2025 Kevin AI | Powered by OpenAI & Elasticsearch</div>", unsafe_allow_html=True)
