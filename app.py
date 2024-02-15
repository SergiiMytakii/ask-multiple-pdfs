import streamlit as st
import os
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.prompts import  ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter

lawyer_vectorstore = 'vectorstore.pkl'
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print(f"Number of chunks: {len(chunks)}")
    return chunks

def get_saved_vectorstore(save_path):
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            vectorstore = pickle.load(f)
        print(f"use saved Vectorstore ")
        return vectorstore
    else:
        return None


def get_vectorstore(text_chunks, save_path=None):
    if save_path and os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            vectorstore = pickle.load(f)
    else:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(vectorstore, f)
    print(f"Vectorstore is ready")
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    # llm = HuggingFaceHub(repo_id="google-bert/bert-base-uncased", task="text2text-generation")

    
    memory = ConversationSummaryBufferMemory(
        max_token_limit=1000,
        llm=llm,
        memory_key='chat_history', return_messages=True)
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})


    embeddings = OpenAIEmbeddings()
    splitter = CharacterTextSplitter(chunk_size=334, chunk_overlap=0, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings, similarity_threshold=0.76)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[ splitter,
            redundant_filter, relevant_filter,  ]
    )
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, base_retriever=retriever
)


    template = """
    1. You are the lawyer consultant. 
    2. Talk like a lawyer, but with simple words.
    3. Use the following pieces of context to answer the question at the end.
    4. Use three sentences maximum and keep the answer as concise as possible.
    5. always ask a questions to get more context.
    6. If you don't know the answer, try to find it your knowledge data about Ukranian law.
    7. Greet the client in the first sentence.
    8. Answer only in Ukrainian.
    9. if you not sure, do not make up any law. Suggest to ask a lawyer.
    10. treat letter "ґ" as "г".

    context: {context}

    Question: {question}

    Helpful Answer:"""


    prompt = ChatPromptTemplate.from_template( template )




    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever,
        memory=memory,
        verbose= True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return conversation_chain


def handle_userinput(user_question):

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    st.write(response)
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
            vectorstore = get_saved_vectorstore(lawyer_vectorstore)
            
            if vectorstore is not None:
                if st.session_state.conversation is None:
                    st.session_state.conversation = get_conversation_chain(
                            vectorstore)
                show_retriver_results(user_question, vectorstore)
                handle_userinput(user_question)
            else: 
                st.write("Please upload your documents first")

    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks, save_path=lawyer_vectorstore)


                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

def show_retriver_results(user_question, vectorstore):
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    compressor = LLMChainFilter.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )
    retriever_results = compression_retriever.get_relevant_documents(user_question)
    pretty_print_docs(retriever_results)

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )



if __name__ == '__main__':
    main()
