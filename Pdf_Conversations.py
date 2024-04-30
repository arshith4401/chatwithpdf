import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
os.environ['GOOGLE_API_KEY'] = "AIzaSyA5P7oBQdm_FCswOWax9LyQpI4db4DTJTA"

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def extract_text_from_pdfs(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def setup_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def process_user_input(user_questions, pdfs):
    if not pdfs:
        st.error("No PDF documents uploaded.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    for user_question in user_questions:
        st.write(f"Question: {user_question}")
        st.write("Context:")
        for pdf in pdfs:
            if pdf:
                text = pdf[:200] if len(pdf) > 200 else pdf  # Display first 200 characters of each document
                st.text(text)
            else:
                st.text("No content")
        document_vectors = vector_store.similarity_search(user_question)
        conversational_chain = setup_conversational_chain()
        try:
            response = conversational_chain({"input_documents": document_vectors, "question": user_question}, return_only_outputs=True)
            st.write("Reply:", response["output_text"])
        except Exception as e:
            st.error(f"Error: {str(e)}")
        st.markdown("---")

def main():
    st.set_page_config("PDF Chat")
    st.header("PDF Conversations: Explore and Interact")
    user_questions = st.text_input("Ask Questions from the PDF Files (Separate questions with commas)")
    user_questions = [q.strip() for q in user_questions.split(",")] if user_questions else []
    pdfs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    if pdfs:
        with st.sidebar:
            st.title("Menu:")
            if st.button("Process PDFs"):
                with st.spinner("Processing..."):
                    raw_text = extract_text_from_pdfs(pdfs)
                    text_chunks = split_text_into_chunks(raw_text)
                    create_vector_store(text_chunks)
                    st.success("Done")

    if user_questions and pdfs:
        process_user_input(user_questions, [pdf.getvalue() for pdf in pdfs])

    # Save session
    if st.button("Save Session"):
        session_state = {"user_questions": user_questions, "pdfs": pdfs}
        st.session_state["saved_session"] = session_state

    # Load session
    if "saved_session" in st.session_state:
        if st.button("Load Session"):
            session_state = st.session_state["saved_session"]
            process_user_input(session_state["user_questions"], [pdf.getvalue() for pdf in session_state["pdfs"]])

    # Feedback mechanism
    feedback = st.text_area("Feedback on responses (optional)")
    if feedback:
        st.write("Thank you for your feedback!")

if __name__ == "__main__":
    main()
