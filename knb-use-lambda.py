# handler.py
import json
import os
import boto3

from langchain_community.chat_models import BedrockChat
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain.prompts import ChatPromptTemplate


def serialize_response(response):
    return {
        "content": response.content,
        "response_metadata": response.response_metadata,
        "id": response.id,
    }


def bedrock_knowledge_base_handler(event, context):
    # AWS 세션을 설정
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

    # event에서 user_message 추출
    body = json.loads(event["body"])
    user_message = body["user_message"]

    # 유저 메시지에 대해 검색
    docs = retriever.get_relevant_documents(query=user_message)
    docs_content = "\n\n".join(document.page_content for document in docs)

    # ChatPromptTemplate로 prompt를 구성
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.\n\nContext: {context}",
            ),
            ("human", "{question}"),
        ]
    ).format_messages(context=docs_content, question=user_message)

    # LLM에 전달
    response = bedrock_chat.invoke(prompt)
    print(type(response))
    print(response)
    response_data = serialize_response(response)

    # 답변을 리턴하는 부분
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"response": response_data}, ensure_ascii=False),
    }
