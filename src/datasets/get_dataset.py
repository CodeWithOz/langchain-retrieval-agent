import os
import datetime
from dotenv import load_dotenv
from pinecone_datasets import load_dataset


def get_current_timestamp():
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime("%H:%M:%S")


def get_current_timestamp_prefix():
    return f"{get_current_timestamp()} - "


def timed_print(msg: str):
    print(f"{get_current_timestamp_prefix()}{msg}")


load_dotenv()


timed_print(f"loading squad dataset")
dataset = load_dataset("squad-text-embedding-ada-002")
timed_print(f"loaded squad dataset")

timed_print(f"dropping sparse_values and blob columns")
dataset.documents.drop(["sparse_values", "blob"], axis=1, inplace=True)
timed_print(f"dropped sparse_values and blob columns")

# save the dataset locally
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "squad-dataset")
timed_print(f"saving dataset locally")
dataset.to_path(file_path)
timed_print(f"saved dataset locally")
