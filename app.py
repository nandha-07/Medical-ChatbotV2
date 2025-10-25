from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# make sure this returns a LangChain-compatible Embeddings instance
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# initialize Google/Gemini model
chatModel = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# IMPORTANT: prompt must include the "context" variable (used by create_stuff_documents_chain)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful medical chatbot."),
        # include context and user question (input)
        ("human", "Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{input}\n\nAnswer:")
    ]
)

# create the combine/stuff chain (llm + prompt)
question_answer_chain = create_stuff_documents_chain(llm=chatModel, prompt=prompt)

# create retrieval (RAG) chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User input:", msg)
    # create_retrieval_chain expects {"input": <question>} when invoking
    response = rag_chain.invoke({"input": msg})
    answer = response.get("answer") if isinstance(response, dict) else str(response)
    print("Response:", answer)
    return str(answer)


if __name__ == "__main__":
    # keep debug True only during development
    app.run(host="0.0.0.0", port=8080, debug=True)
