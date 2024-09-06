import os
from dotenv import load_dotenv
load_dotenv()

import lancedb
from lancedb.rerankers import CohereReranker

uri = "./db"
db = lancedb.connect(uri)
table = db.open_table("artisan")
reranker = CohereReranker()

query = "What can Ava do?"
docs = table.search(query, query_type="hybrid").limit(5).rerank(reranker=reranker).to_pandas()["text"].to_list()
print(docs)