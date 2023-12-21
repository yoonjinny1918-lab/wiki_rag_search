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


st.title("Kevin의 위키피디아 AI 검색기(RAG)")
st.caption("Semantic search and Retrieval augmented generation using Elasticsearch and OpenAI

            데이터 출처
            - https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip
            
            시스템 구현 방식
            - OpenAI Wikipedia 벡터 데이터 세트를 Elasticsearch로 색인하기
            - OpenAI 임베딩 엔드포인트로 질문 임베딩하기
            - 인코딩된 질문을 사용해 Elasticsearch 색인에서 시맨틱 검색(KNN)을 수행합니다.
            - 검색 증강 생성(RAG)을 위해 상위 검색 결과를 OpenAI 채팅 완성 API 엔드포인트로 보내기
           ")

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
        st.write(choice.message.content)
