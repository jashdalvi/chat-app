import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import LanceDB
from langchain.embeddings.openai import OpenAIEmbeddings
import lancedb
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import Vector, LanceModel
from langchain.docstore.document import Document
load_dotenv()

openai = get_registry().get("openai").create()
class Schema(LanceModel):
    text: str = openai.SourceField()
    vector: Vector(openai.ndims()) = openai.VectorField()


embedding_function = OpenAIEmbeddings()

db = lancedb.connect("./db")
table = db.create_table(
    "artisan",
    schema=Schema,
    mode="overwrite",
)

def find_headers_markdown(markdown_text):
    # Define a regular expression to match Markdown headers
    header_regex = re.compile(r'^((#{1,6}\s+.*$)|(.*[\r\n](=+|-+)[\r\n]))', re.MULTILINE)

    # Use the regex to find all headers in the text
    headers = []
    for match in header_regex.finditer(markdown_text):
        full_header = match.group(1)
        if match.group(2):
            # Match headers in the form of "### Header"
            level = len(match.group(2).strip().split(" ")[0])
            text = match.group(2).strip().split(" ", 1)[1]
            headers.append({
                'level': level,
                'text': text,
                'full_header': full_header.strip(),
            })
        elif match.group(3):
            # Match headers in the form of "Header\n=====" or "Header\n-----"
            level = 1 if match.group(3).endswith("=") else 2
            text = match.group(3).strip().split('\n')[0]
            headers.append({
                'level': level,
                'text': text,
                'full_header': full_header.strip(),
            })

    return headers
with open("./data/results.json", "r") as file:
    results = json.load(file)



final_docs = []
for r in results:
    cleaned_markdown = re.sub(r'!\[.*?\]\((.*?)\)', '', r['markdown'])
    separators = [h["full_header"] for h in find_headers_markdown(cleaned_markdown)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128, separators=separators)
    docs = text_splitter.split_text(cleaned_markdown)
    final_docs.extend([Document(page_content=d) for d in docs if len(d) > 50])

db = LanceDB.from_documents(final_docs, embedding_function, connection=table)
table.create_fts_index("text")
table.to_pandas().head()
