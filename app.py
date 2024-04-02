from connect import bedrock_client
from langchain_community.chat_models import BedrockChat
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI

gpt_llm = ChatOpenAI(temperature=0.1)

bedrock_chat = BedrockChat(
    client=bedrock_client,
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={
        "temperature": 0.1,
    },
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Translate this sentence from {lang_a} to {lang_b}: {sentence}",
        ),
    ]
)

chain = prompt | bedrock_chat

result = chain.invoke(
    {
        "lang_a": "English",
        "lang_b": "Korean",
        "sentence": "I love amazon!",
    }
)

print(result)
