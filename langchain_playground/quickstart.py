# https://python.langchain.com/v0.1/docs/use_cases/chatbots/quickstart/

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory

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
