from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from flask import Flask, render_template, request, jsonify
import os
import constants

app = Flask(__name__)

os.environ["OPENAI_API_TYPE"] = constants.APITYPE
os.environ["AZURE_OPENAI_API_VERSION"] = constants.APIVERSION
os.environ["AZURE_OPENAI_ENDPOINT"] = constants.APIBASE
os.environ["AZURE_OPENAI_API_KEY"] = constants.APIKEY

auth_headers = {
    "Ocp-Apim-Subscription-Key": constants.APIKEY,
    "x-service-line": constants.XSERVICELINE,
    "x-brand": constants.XBRAND,
    "x-project": constants.XPROJECT,
    "api-version": constants.APIVERSION,
    "Content-Type": "application/json",
    "Cache-Control": "no-cache",
}
embeddings = AzureOpenAIEmbeddings(
      api_key=constants.APIKEY,
      azure_endpoint=constants.APIBASE,
      model="text-embedding-ada-002",
      azure_deployment="TextEmbeddingAda2",
      api_version=constants.API_VERSION,
      default_headers=auth_headers,
   )
llm = AzureChatOpenAI(
      api_key=constants.APIKEY,
      azure_endpoint=constants.APIBASE,
      model="GPT35Turbo",
      azure_deployment=constants.DEPLOYMENT,
      api_version=constants.API_VERSION,
      default_headers=auth_headers,
   )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    results = search_vectorstore(query)
    return jsonify(results['result'])
    
def load_data_vectorstore():
    # Define a custom metadata function
    def extract_metadata(record,metadata):
        metadata["name"] = record.get("name")
        metadata["courses"] = record.get("courses")
        metadata["admission_process"] = record.get("admission_process")
        metadata["teacher_profiles"] = record.get("teacher_profiles")
        metadata["alumni_info"] = record.get("alumni_info")
        metadata["career_paths"] = record.get("career_paths")
        return metadata

    # Initialize an empty list to hold all documents
    all_documents = []

    # Read all json files using JSONLoader
    for filename in os.listdir('.'):
        if filename.endswith('.json'):
            loader = JSONLoader(filename, jq_schema='.', metadata_func=extract_metadata, text_content=False)
            documents = loader.load()
            all_documents.extend(documents)
    
    # Store the document in vector store (in memory)
    global vectorstore
    vectorstore = FAISS.from_documents(all_documents, embeddings)

def search_vectorstore(query):
    # Retrieve the results from vector store based on query and use llm to formulate response
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain.invoke(query)

if __name__ == '__main__':
    load_data_vectorstore()
    app.run(debug=True)
