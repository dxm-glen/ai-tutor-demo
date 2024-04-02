import streamlit as st
import os
import boto3
from langchain_community.chat_models import BedrockChat
from langchain_openai import ChatOpenAI


gpt_chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    # callbacks=[ChatCallbackHandler()],
)

session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
)

bedrock_client = session.client("bedrock-runtime", "us-east-1")

bedrock_chat = BedrockChat(
    client=bedrock_client,
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    streaming=True,
    # callbacks=[ChatCallbackHandler()],
    model_kwargs={
        "temperature": 0.1,
    },
)
