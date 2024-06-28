# Install Ollama from website, look at models available on website llama3
# $ ollama pull llama3 -> downloads llm locally to computer
# $ ollama serve
# $ ollama run llama3 -> to test that it's working

# $ mkdir Rag
# $ cd Rag
# $ python3 -m venv venv
# $ source venv/bin/activates

# langchain is a framework designed to simplify the creation of applications using LLMs
# Ollama llama3 set up with langchain
from flask import Flask, request
from flask_cors import CORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

# $ pip install flask
# $ pip install langchain-community
# $ pip install langchain-text-splitters
# $ pip install fastembed
# $ pip3 install pdfplumber
# $ pip3 install chromadb
# $ pip3 install -U langchain langchain-community

app = Flask(__name__)
CORS(app)

folder_path = "db"

llm = Ollama(model="llama3")

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template("""
    <s>[INST] You answer questions and search documents. If you do not have an answer from provided information, say so. [/INST]</s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
""")

@app.route("/ai", methods=["POST"])
def aiPost():
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")
    response = llm.invoke(query)
    print(response)
    answer = {"answer": response}
    return answer

@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "files/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")
    docs = PDFPlumberLoader(save_file).load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
        )
    vector_store.persist()
    
    reponse = {"status": "success", "filename": file_name}
    return reponse

@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")
    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )
    document_chain = create_stuff_documents_chain(llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    response = chain.invoke({"input": query})
    print(response)
    sources = []
    for doc in response["context"]:
        sources.append(
            {"Source": doc.metadata["source"], "page_content": doc.page_content}
        )
    answer = {"answer": response["answer"], "sources": sources}
    return answer

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)