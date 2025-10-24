import os
import pandas as pd
import streamlit as st
from openai import OpenAI
from elasticsearch import Elasticsearch

# 🎯 페이지 설정
st.set_page_config(
    page_title="한글 위키 Q&A AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 🔐 API 키 로드
client = OpenAI(api_key=st.secrets["api_key"])
ELASTIC_CLOUD_ID = st.secrets["elastic_cloud_key"]
ELASTIC_API_KEY = st.secrets["elastic_api_key"]

# 🔍 Elasticsearch 연결
es = Elasticsearch(
  cloud_id=ELASTIC_CLOUD_ID,
  api_key=ELASTIC_API_KEY
)

# ✅ 연결 테스트
try:
    es.info()
except Exception as e:
    st.error("🚨 Elasticsearch 연결 실패: " + str(e))

# ------------------------------
# 🎨 헤더
# ------------------------------
st.markdown("""
<div style='text-align: center;'>
  <h1>🤖 한글로 답변하는 AI</h1>
  <h4 style='color:gray;'>Semantic Search + RAG (Elasticsearch + OpenAI)</h4>
  <p>영문 위키피디아를 기반으로 질문에 대한 답변을 제공합니다.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# 🧭 정보 섹션
# ------------------------------
with st.expander("ℹ️ 서비스 설명 보기", expanded=False):
    st.markdown("""
    **추천 질문 예시**
    - 🌊 대서양은 몇 번째로 큰 바다인가?
    - 🏙️ 대한민국의 수도는?
    - ⚔️ 이순신의 출생년도는?
    - 🚗 도요타에서 가장 많이 팔리는 차는?

    **데이터 출처**
    - [Wikipedia Vector Dataset (OpenAI)](https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip)
    - 데이터 수: 약 25,000건  
    - 설명: 영어 위키피디아 문서 기반 벡터 임베딩 검색
    """)

# ------------------------------
# 💬 입력 폼
# ------------------------------
with st.container():
    st.markdown("### 🧠 질문 입력")
    with st.form("form"):
        question = st.text_input("궁금한 내용을 입력하세요 💬", placeholder="예: 태양은 몇 번째로 큰 별인가요?")
        submit = st.form_submit_button("🔍 검색 및 답변 생성")

# ------------------------------
# 🚀 처리 및 결과 출력
# ------------------------------
if submit and question:
    with st.spinner("🤔 Kevin AI가 답변을 생성 중입니다..."):
        print("질문:", question)

        # 1️⃣ 질문 번역
        translation = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Translate the following Korean text to English:\n{question}"}
            ]
        )
        translated_question = translation.choices[0].message.content
        print("번역:", translated_question)

        # 2️⃣ 임베딩 생성
        question_embedding = client.embeddings.create(
            input=[translated_question],
            model="text-embedding-ada-002"
        ).data[0].embedding

        # 3️⃣ Elasticsearch 검색
        response = es.search(
            index="wikipedia_vector_index",
            knn={
                "field": "content_vector",
                "query_vector": question_embedding,
                "k": 10,
                "num_candidates": 100
            }
        )

        top_hit = response['hits']['hits'][0]['_source']
        top_hit_summary = top_hit['text']

        # 4️⃣ OpenAI를 통한 답변 생성
        summary = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant for question-answering tasks. Use the provided context to answer the question. If unsure, say you don't know."},
                {"role": "user", "content": f"Answer in Korean using up to three sentences.\nQuestion: {translated_question}\nContext: {top_hit_summary}"}
            ]
        )

        answer = summary.choices[0].message.content

    # ------------------------------
    # 🧾 결과 출력
    # ------------------------------
    st.success("✅ AI의 답변:")
    st.markdown(f"<div style='background-color:#f9f9f9;padding:15px;border-radius:10px;'>{answer}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📚 검색된 문서 목록")

    # 📄 카드 형태로 문서 리스트 표시
    for hit in response['hits']['hits']:
        title = hit['_source']['title']
        url = hit['_source']['url']
        score = hit['_score']
        with st.container():
            st.markdown(f"""
            <div style='background-color:#f1f3f6;padding:12px;border-radius:8px;margin-bottom:8px;'>
                <b>📖 <a href='{url}' target='_blank'>{title}</a></b><br>
                <span style='color:gray;'>유사도 점수: {round(score,2)}</span>
            </div>
            """, unsafe_allow_html=True)
