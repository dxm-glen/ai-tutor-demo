import os
import boto3

from langchain_community.chat_models import BedrockChat
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever

from langchain.prompts import ChatPromptTemplate


session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name="us-east-1",
)

bedrock_client = session.client("bedrock-runtime", "us-east-1")

bedrock_chat = BedrockChat(
    client=bedrock_client,
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={
        "temperature": 0.1,
    },
)

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="XPMNH9ZTJI",
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 5}},
)

# 유저 질문을 받아서 넣을 것
# query = "졸업 관련 정보를 알려주세요"
query = "화이트와인에 대해서 알려주세요"

# 유저 질문에 대해 검색
# 검색된 문서들의 내용을 결합
docs = retriever.get_relevant_documents(query=query)
docs_content = "\n\n".join(document.page_content for document in docs)
print(docs_content)
print("------------------------------------------------------------------------")

# ChatPromptTemplate로 prompt를 구성
# 검색해온 context와 질문을 전달
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.\n\nContext: {context}",
        ),
        ("human", "{question}"),
    ]
).format_messages(context=docs_content, question=query)
print(prompt)
print("------------------------------------------------------------------------")

# LLM에 전달
# response = bedrock_chat(prompt)
response = bedrock_chat.invoke(prompt)

# 답변을 출력, 나중엔 클라이언트에 리턴

print(response)
print("------------------------------------------------------------------------")
