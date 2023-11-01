from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.openai import OpenAI

load_dotenv()

def main():
    st.set_page_config(page_title="Query the PDF streamlit app")
    st.header(body="Query the PDF 💬")
    pdf = st.file_uploader(label="Upload your pdf file", type="pdf")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
        
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
        )

        chunks = text_splitter.split_text(pdf_text)
        emeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        knowledge_base = FAISS.from_texts(chunks, embedding=emeddings)

        user_question = st.text_input(label="Ask your question")

        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
            st.write(response)


if __name__ == '__main__':
    main()