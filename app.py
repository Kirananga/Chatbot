import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from googletrans import Translator
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    if the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer.
    
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_summarization_chain():
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = load_summarize_chain(model, chain_type="map_reduce")
    return chain

def translate_input(user_question):
    translator = Translator()
    try:
        translated = translator.translate(user_question, src='auto', dest='en')
        st.write(f"Translated Question: {translated.text} (Detected language: {translated.src})")
        return translated.text
    except Exception as e:
        st.error("Translation failed. Please check your input or try again later.")
        print(f"Translation Error: {e}")
        return user_question 

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    translated_question = translate_input(user_question)

    docs = new_db.similarity_search(translated_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": translated_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])

def summarize_pdf():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search("Summarize the content")
        chain = get_summarization_chain()
        response = chain({"input_documents": docs}, return_only_outputs=True)
        st.write("Summary of the PDF Content:")
        st.write(response["output_text"])
    except Exception as e:
        st.error("Error in summarization. Please try again.")
        print(f"Summarization Error: {e}")

def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("KIRAN BOT")
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done! The PDFs are now ready for queries.")
        
        if st.button("Summarize PDF Content"):
            with st.spinner("Summarizing..."):
                summarize_pdf()

if __name__ == "__main__":
    main()
