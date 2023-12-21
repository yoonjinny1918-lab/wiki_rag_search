import os
import pandas as pd
import streamlit as st
from openai import OpenAI
from elasticsearch import Elasticsearch

client = OpenAI(api_key=st.secrets["api_key"])

# https://www.elastic.co/search-labs/tutorials/install-elasticsearch/elastic-cloud#finding-your-cloud-id
ELASTIC_CLOUD_ID = st.secrets["elastic_cloud_key"]

# https://www.elastic.co/search-labs/tutorials/install-elasticsearch/elastic-cloud#creating-an-api-key
ELASTIC_API_KEY = st.secrets["elastic_api_key"]

es = Elasticsearch(
  cloud_id = ELASTIC_CLOUD_ID,
  api_key=ELASTIC_API_KEY
)

# Test connection to Elasticsearch
print(es.info())


st.title("영문 위키피디아 기반, 한글로 답변하는 AI")
st.subheader("Semantic search and Retrieval augmented generation using Elasticsearch and OpenAI")

st.caption('''
졸은 질문 예 : 
- How big is the Atlantic ocean?
- 대한민국의 수도는?
- 이순신의 출생년도는?
- 북한과 남한의 대표적인 장단점을 3가지만 정리해줘.

장점
- 전문 영어 문서(예: 해외 기업투자분석리포트 등)를 대상으로 한글로 검색하고 답변 받기 용이합니다.
- 전통적인 검색방식(키워드 매칭)을 탈피하여 의미가 유사한 문서를 찾고 이를 통한 답변 가능
- 일반 검색에 비하여 2~3 단어의 이상의 문장형 검색 질의에 답변 잘함
- 다국어(multi-lingual AI) 검색 가능(단, 학습문서가 영어라서 영어로 질의해야 답변 잘해요.)
- LLM(Large Language Model) AI의 답변이 아닌 내가 원하는 문서를 통해서 답변 가능하여 할루시네이션(Hallucination)을 최소화
- Private한 사내 문서나 특정 도메인에 특화된 답변이 가능합니다.(사내 문서가 외부로 노출되지 않음)

단점
- 데이터가 충분하지 않아 다양한 질문에 정확한 답을 못할 수 있음.
- 유사도 검색의 특성상 적절하지 않은 검색결과가 있다면 잘못된 답변을 할 수 있음

데이터 출처
- https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip
- 데이터 설명 : https://weaviate.io/developers/weaviate/tutorials/wikipedia
- 데이터 건수 : 25,000건 (데이터의 양을 늘리면, 다양한 질문에 대한 답변 가능)

시스템 구현 방식
- OpenAI Wikipedia 벡터 데이터 세트를 Elasticsearch(검색엔진)로 색인
- OpenAI Embedding을 통하여 사용자 질문 임베딩
- 임베딩된 질문을 사용해 Elasticsearch에서 시맨틱 검색(KNN : 벡터 유사도)을 수행
- 검색 증강 생성(RAG)을 위해 상위 검색 결과를 이용하여 OpenAI 채팅 완성 API를 사용하여 요약하여 답변
''')

with st.form("form"):
    question = st.text_input("Prompt")
    submit = st.form_submit_button("Submit")

if submit and question:
  with st.spinner("Waiting for Kevin AI..."):
      question = question.replace("\n", " ")
      question_embedding = client.embeddings.create(input = [question], model="text-embedding-ada-002").data[0].embedding
    
      response = es.search(
        index = "wikipedia_vector_index",
        knn={
            "field": "content_vector",
            "query_vector":  question_embedding,
            "k": 10,
            "num_candidates": 100
          }
      )
      
      top_hit_summary = response['hits']['hits'][0]['_source']['text'] # Store content of top hit for final step
      
      summary = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": "Answer the following question in Korean.:"
               + question
               + "by using the following text:"
               + top_hit_summary},
          ]
      )
  
      choices = summary.choices
    
      for choice in choices:
        print(choice.message.content)
        st.markdown(choice.message.content)
