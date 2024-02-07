import os
from pinecone import Pinecone, PodSpec
from pinecone_datasets import Dataset
from dotenv import load_dotenv
import datetime


def get_current_timestamp():
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime("%H:%M:%S")


load_dotenv()


def get_current_timestamp_prefix():
    return f"{get_current_timestamp()} - "


def timed_print(msg: str):
    print(f"{get_current_timestamp_prefix()}{msg}")


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
    pc.create_index(
        name=index_name,
        spec=spec,
        dimension=1536,
        metric="dotproduct"
    )
    timed_print(f"Created index {index_name!r}")

index = pc.Index(index_name)
timed_print(f"Index stats: {index.describe_index_stats()}")

if index.describe_index_stats().get("total_vector_count", 0) == 0:
    timed_print(f"Loading squad dataset")
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "squad-dataset")
    dataset = Dataset.from_path(file_path)
    timed_print(f"Loaded squad dataset")
    timed_print(f"Populating index with dataset")
    index.upsert_from_dataframe(dataset.documents, batch_size=100)
    timed_print(f"Populated index with dataset: {index.describe_index_stats()}")
else:
    timed_print(f"Index {index!r} contains {index.describe_index_stats().get('total_vector_count', 0)} vectors")
