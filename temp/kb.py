from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter

# Load documents from the directory
loader = DirectoryLoader('documents')
documents = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=128, separator=r'\n| ', is_separator_regex=True)
split_documents = text_splitter.split_documents(documents)

# Display details of the split documents
total_chunks = len(split_documents)
print(f"Total number of chunks: {total_chunks}")

for i, chunk in enumerate(split_documents):
    chunk_size = len(chunk.page_content)
    print(f"Chunk {i+1}: Size = {chunk_size} characters")
