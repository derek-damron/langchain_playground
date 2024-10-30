# https://python.langchain.com/v0.1/docs/use_cases/chatbots/quickstart/
# exec(open('quickstart.py').read())

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import Dict
from langchain_core.runnables import RunnablePassthrough

#####
# Setup
#

load_dotenv()
chat = ChatAnthropic(model="claude-3-5-sonnet-20240620")

#####
# Sending and receiving a message
#

first_message = HumanMessage(
    content="Translate this sentence from English to French: I love programming."
)

first_response = chat.invoke([first_message])

print(first_response.content)

#####
# Conversation chain
#

second_message = HumanMessage(content="What did you just say?")

second_response = chat.invoke(
    [first_message,
        first_response,
        second_message,
    ]
)

print(second_response.content)

#####
# Prompt templates
#

system_template = ("system", "You are a helpful assistant. Answer all questions to the best of your ability.")

prompt = ChatPromptTemplate.from_messages(
    [system_template,
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | chat

chain_response = chain.invoke(
    {
        "messages": [first_message,
            first_response,
            second_message,
        ],
    }
)

print(chain_response.content)

#####
# Message history
#

chat_history = ChatMessageHistory()
chat_history.add_user_message("hi!")
chat_history.add_ai_message("whats up?")

print(chat_history.messages)

chat_history.add_user_message(
    "Translate this sentence from English to French: I love programming."
)

history_response = chain.invoke({"messages": chat_history.messages})

print(history_response.content)

chat_history.add_ai_message(history_response)
chat_history.add_user_message("What did you just say?")

second_history_response = chain.invoke({"messages": chat_history.messages})

print(second_history_response.content)

#####
# Retrievers
#

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OllamaEmbeddings(model="llama3"))

retriever = vectorstore.as_retriever(k=4)

docs = retriever.invoke("how can langsmith help with testing?")

chat_docs = ChatAnthropic(model="claude-3-5-sonnet-20240620")

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = create_stuff_documents_chain(chat_docs, question_answering_prompt)

demo_ephemeral_chat_history = ChatMessageHistory()

demo_ephemeral_chat_history.add_user_message("how can langsmith help with testing?")

document_chain_response = document_chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
        "context": docs,
    }
)

print(document_chain_response)

#####
# Retrieval chain
#

def parse_retriever_input(params: Dict):
    return params["messages"][-1].content

retrieval_chain = RunnablePassthrough.assign(
    context=parse_retriever_input | retriever,
).assign(
    answer=document_chain,
)

response = retrieval_chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
    }
)

print(response)

demo_ephemeral_chat_history.add_ai_message(response["answer"])

demo_ephemeral_chat_history.add_user_message("Can you elaborate more on performance monitoring with langsmith?")

response2 = retrieval_chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
    },
)

print(response2['answer'])

# If we want to pass in only the answer

retrieval_chain_with_only_answer = (
    RunnablePassthrough.assign(
        context=parse_retriever_input | retriever,
    )
    | document_chain
)

retrieval_chain_with_only_answer.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
    },
)