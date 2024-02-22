import streamlit as st
import os
import pickle
from dotenv import load_dotenv
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import  ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter
from langchain.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from summarizer import Summarizer


doc_vectorstore = 'vectorstore.pkl'
def get_file_path(pdf_docs):
    if pdf_docs is not None:
        temp_pdfs_path = []
        for file in pdf_docs:
            temp_dir = st.session_state.get('temp_dir', os.path.join(os.getcwd(), 'temp'))
            os.makedirs(temp_dir, exist_ok=True) 
            temp_file_path = os.path.join(temp_dir, file.name)
            temp_pdfs_path.append(temp_file_path)
            # Save the file temporarily
            with open(temp_file_path, "wb") as f:  
                f.write(file.getbuffer())
            print("File saved temporarily at:", temp_file_path)
            st.session_state.pdfs_dir = temp_dir
        return temp_pdfs_path


def get_pdf_text(pdf_docs ):
    temp_pdfs_path = get_file_path(pdf_docs)
    docs = []
    for pdf in temp_pdfs_path:
        pdf_reader = PyPDFLoader(pdf)
        docs.extend(pdf_reader.load())
    return docs


def get_text_chunks(docs: list):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=0,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Number of chunks: {len(chunks)}")
    return chunks



def get_vectorstore(text_chunks, save_path=None):
   
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=text_chunks, embedding=embeddings, persist_directory="./data/vectorstore")
    print(f"Vectorstore is ready")
    return vectorstore


def get_conversation_chain(vectorstore: Chroma):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, verbose=True)
    # llm = HuggingFaceHub(repo_id="google-bert/bert-base-uncased", task="text2text-generation")

    
    memory = ConversationSummaryBufferMemory(
        max_token_limit=1000,
        llm=llm,
        memory_key='chat_history', return_messages=True)
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    #retriever=  compressor_retriever(retriever)
    template = """
    1. You are the expert document analizer. 
    2. if user asks to summarize the document, answer only: "Summary of the document:"
    4. Use the following pieces of context to answer the question at the end.
    5. If you don't know the answer, say "I don't know, I can only edit the document".


    context: {context}

    Question: {question}

    Helpful Answer:"""

    prompt = ChatPromptTemplate.from_template( template )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose= True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return conversation_chain

def compressor_retriever(retriever):
    embeddings = OpenAIEmbeddings()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings, similarity_threshold=0.76)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[ splitter,
            redundant_filter, relevant_filter,  ]
    )
    compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)
    return compression_retriever



def handle_userinput(user_question):

    response = st.session_state.conversation({'question': user_question})
    print(st.session_state.pdfs_dir)
    if "Summary of the document" in response["answer"]:
        st.write("Summary of the document...")
        summary  =  Summarizer().summarize_the_pdf(
            file_dir=st.session_state.pdfs_dir,
            max_final_token=3500,
            token_threshold=0,
            temperature=0,
            gpt_model="gpt-3.5-turbo",
            character_overlap=100)
        st.write(summary)
    else:    

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
        handle_userinput(user_question)
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True,)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # get the text chunks
                docs_chunks = get_text_chunks(raw_text)
                # create vector store
                vectorstore = get_vectorstore(docs_chunks, save_path=doc_vectorstore)
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

def show_retriver_results(user_question, vectorstore):
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    retriever_results = retriever.get_relevant_documents(user_question)
    pretty_print_docs(retriever_results)

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )



if __name__ == '__main__':
    main()
