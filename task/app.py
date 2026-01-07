from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


# TODO:
# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope
SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Reference specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

# TODO:
# Provide structured system prompt, with RAG Context and User Question sections.
USER_PROMPT = """##RAG CONTEXT:
{context}

##USER QUESTION: 
{query}
"""


# TODO:
# - create embeddings client with 'text-embedding-3-small-1' model
# - create chat completion client
# - create text processor, DB config: {'host': 'localhost','port': 5433,'database': 'vectordb','user': 'postgres','password': 'postgres'}
# ---
# Create method that will run console chat with such steps:
# - get user input from console
# - retrieve context
# - perform augmentation
# - perform generation
# - it should run in `while` loop (since it is console chat)

embeddings_client = DialEmbeddingsClient(
    deployment_name="text-embedding-3-small-1", api_key=API_KEY
)
chat_completion_client = DialChatCompletionClient(
    deployment_name="gpt-4o", api_key=API_KEY
)

text_processor = TextProcessor(
    embeddings_client=embeddings_client,
    db_config={
        "host": "localhost",
        "port": 5433,
        "database": "vectordb",
        "user": "postgres",
        "password": "postgres",
    },
)


def main():
    text_processor.process_text_file(
        file_name="microwave_manual.txt",
        chunk_size=400,
        overlap=40,
        dimensions=384,
        truncate_table=True,
    )
    print("\n Context was loaded to VectorDB")

    conversation = Conversation()
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))

    while True:
        user_input = input("\n➡️ ").strip()

        if user_input.lower().strip() in ["quit", "exit"]:
            print("👋 Goodbye")
            break

        print("\n🔍 STEP 1: RETRIEVAL\n")
        context = text_processor.search(
            search_mode=SearchMode.EUCLIDIAN_DISTANCE,
            user_request=user_input,
            top_k=5,
            score_threshold=0.03,
            dimensions=384,
        )

        # Step 2: Augmentation
        print("\n🔗 STEP 2: AUGMENTATION\n")
        augmented_prompt = USER_PROMPT.format(
            context="\n\n".join(context), query=user_input
        )
        conversation.add_message(Message(Role.USER, augmented_prompt))
        print(f"Prompt:\n{augmented_prompt}")

        # Step 3: Generation
        print("\n🤖 STEP 3: GENERATION\n")
        ai_message = chat_completion_client.get_completion(conversation.get_messages())
        print(f"✅ RESPONSE:\n{ai_message.content}")
        print("=" * 100)
        conversation.add_message(ai_message)


main()

# TODO:
#  PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
#  RUN docker-compose.yml
