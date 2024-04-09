import streamlit as st
import os
import boto3
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain.chains import RetrievalQA

from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

st.set_page_config(
    page_title="FAQ",
    page_icon="🆀",
)


class ChatCallbackHandler(StreamingStdOutCallbackHandler):

    message = ""

    def on_chat_model_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


gpt_chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
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
    callbacks=[ChatCallbackHandler()],
    model_kwargs={
        "temperature": 0.1,
    },
)


user_selected_model = st.selectbox(
    "Choose your MODEL (Default is AWS Bedrock)",
    (
        "BedRock",
        "GPT",
    ),
)
if user_selected_model == "GPT":
    selected_model = gpt_chat
else:
    selected_model = bedrock_chat


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read().decode("utf-8")
    file_path = f"./.cache/files/{file.name}"

    with open(file_path, mode="wb") as f:
        f.write(file_content.encode("utf-8"))
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    txt_loader = TextLoader(f"./files/{file.name}")  # 리스트로 나옴
    # pdf_loader = PyPDFLoader("./files/FAQ.pdf")  # 리스트로 나옴 메타데이터에 각 페이지
    docs = txt_loader.load_and_split(text_splitter=splitter)
    # embeddings = OpenAIEmbeddings()
    embeddings = BedrockEmbeddings(
        credentials_profile_name="glen.lee.nxtcloud", region_name="us-east-1"
    )

    # 각 문서에 대해 임베딩을 수행하고 결과를 캐시에 저장
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
    )
    print(cached_embeddings)

    # 임베딩된 문서를 기반으로 FAISS 인덱스 생성
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    # retriever = vectorstore.as_retriever()
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="PUIJP4EQUA",
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
    )
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def show_messages_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Conext: {context}
        """,
        ),
        ("human", "{question}"),
    ]
)

st.title("FAQ")

st.markdown(
    """
    Welcome!
    Use this chatbot to ask Questions to an AI about FAQ

    upload your file on sidebar
"""
)
with st.sidebar:
    file = st.file_uploader("Upload a .txt, .pdf", type=["pdf", "txt"])
if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask me!", "ai", save=False)
    show_messages_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | selected_model
        )
        with st.chat_message("ai"):
            chain.invoke(
                message
            )  # 여기서 전달되는 message가 RunnablePassthrough로 가는 것

        # 위 체인이 아래 수동 작업을 자동화
        # docs = retriever.invoke(message)
        # docs = "\n\n".join(document.page_content for document in docs)
        # prompt = prompt.format_messages(
        #     context=docs,
        #     question=message,
        # )

else:
    st.session_state["messages"] = []
