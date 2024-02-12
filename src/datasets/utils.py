import datetime
import time
import pinecone
from pinecone_datasets import Dataset


def get_current_timestamp():
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime("%H:%M:%S")


def get_current_timestamp_prefix():
    return f"{get_current_timestamp()} - "


def timed_print(msg: str):
    print(f"{get_current_timestamp_prefix()}{msg}")


def upsert_dataset_redundantly(
    dataset: Dataset, index: pinecone.Index, batch_size: int = 100
):
    timed_print(
        f"upserting {dataset.documents.shape[0]} documents in batches of {batch_size}"
    )
    start_timestamp = get_current_timestamp()
    index_total_vector_count = index.describe_index_stats().get("total_vector_count", 0)
    last_batch = int((index_total_vector_count / batch_size) + 1)
    for i, batch in enumerate(dataset.iter_documents(batch_size=batch_size)):
        if i + 1 < last_batch:
            continue
        timed_print(f"upserting batch {i + 1}")
        try:
            index.upsert(batch)
        except Exception as e:
            timed_print(f"Error during upsert:\n{str(e)}")
            if i + 1 == last_batch:
                # the same batch failed, stop now
                raise e
            # wait a bit then try again
            timed_print(
                f"waiting for 10 seconds before re-attempting upsert of batch {i + 1}"
            )
            time.sleep(10)
            timed_print(f"re-attempting upsert of batch {i + 1}")
            try:
                index.upsert(batch)
            except Exception as e:
                timed_print(f"Error during first re-attempted upsert:\n{str(e)}")
                timed_print(
                    f"waiting for another 10 seconds before re-attempting upsert of batch {i + 1} one more time"
                )
                time.sleep(10)
                timed_print(f"re-attempting upsert of batch {i + 1} one more time")
                index.upsert(batch)
        last_batch = i + 1
        timed_print(f"upserted batch {i + 1}\n")

    end_timestamp = get_current_timestamp()
    timed_print(f"started upserting documents into index at {start_timestamp}")
    timed_print(f"finished upserting documents into index at {end_timestamp}")
