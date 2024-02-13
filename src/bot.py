import os
from langchain.agents import initialize_agent
from langchain_openai import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import pinecone as lc_pinecone
from pinecone import Pinecone, PodSpec
from pinecone_datasets import Dataset
from langchain.agents import Tool
from dotenv import load_dotenv
import datetime

from datasets import utils as datasets_utils


def get_current_timestamp():
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime("%H:%M:%S")


def get_current_timestamp_prefix():
    return f"{get_current_timestamp()} - "


def timed_print(msg: str):
    print(f"{get_current_timestamp_prefix()}{msg}")


load_dotenv()


pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
openai_api_key = os.getenv("OPENAI_API_KEY")


# initialize pinecone
pc = Pinecone(api_key=pinecone_api_key)
spec = PodSpec(environment=pinecone_environment)
index_name = "langchain-retrieval-agent-fast"


# initialize index
if index_name not in pc.list_indexes().names():
    timed_print(f"Creating index {index_name!r}")
    pc.create_index(name=index_name, spec=spec, dimension=1536, metric="dotproduct")
    timed_print(f"Created index {index_name!r}")

index = pc.Index(index_name)
timed_print(f"Index stats: {index.describe_index_stats()}")

if index.describe_index_stats().get("total_vector_count", 0) == 0:
    timed_print(f"Loading squad dataset")
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "datasets", "squad-dataset"
    )
    dataset = Dataset.from_path(file_path)
    timed_print(f"Loaded squad dataset")
    timed_print(f"Populating index with dataset")
    datasets_utils.upsert_dataset_redundantly(dataset, index, 100)
    timed_print(f"Populated index with dataset: {index.describe_index_stats()}")
else:
    timed_print(
        f"Index {index!r} contains {index.describe_index_stats().get('total_vector_count', 0)} vectors"
    )

# creating a vector store and querying
embedding_model_name = "text-embedding-ada-002"
embed = OpenAIEmbeddings(
    model=embedding_model_name,
    api_key=openai_api_key,
)
text_field = "text"
vectorstore = lc_pinecone.Pinecone(index, embed, text_field)
query = "can you tell me some facts about the University of Notre Dame?"

# initialize conversational agent
# chat completion llm
chat_llm_model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name=chat_llm_model_name,
    temperature=0.0,
)
# conversational memory
conversation_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True,
)
# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

tools = [
    Tool(
        name="Knowledge Base",
        func=qa.invoke,
        description=(
            "use this tool when answering general knowledge queries to get "
            "more information about the topic"
        ),
    )
]

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    # verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=conversation_memory,
)

while query:
    print(f"You: {query}")
    print(f"Bot: {agent(query).get('output')}")
    query = input(
        "\n\nI am a general knowledge chatbot 🤖, what would you like to know?\n\n> "
    )
