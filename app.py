#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from flask import Flask, jsonify, request
import os
from constants import CHROMA_SETTINGS

app = Flask(__name__)

# @app.before_first_request
# def setup():
    #run ingest.py and privateGPT.py
load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',2))
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
# activate/deactivate the streaming StdOut callback for LLMs
callbacks = []
# Prepare the LLM
match model_type:
    case "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    case "GPT4All":
        llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    case _default:
        # raise exception if model_type is not supported
        raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
    
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

@app.route("/", methods=["GET"])
def query():
    res = qa(request.headers.get('prompt'))
    docs = res['source_documents']
     # Print the relevant sources used for the answer
    answer = ''
    print(request.headers.get('prompt'))
    for document in docs:
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)
        answer += "\n\n> " + document.metadata["source"] + ":"
        answer += "\n" + document.page_content
    return jsonify({"answer":answer})
if __name__ == '__main__':
    app.run(debug=True)