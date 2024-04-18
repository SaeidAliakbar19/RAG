import streamlit  as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter (
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 100,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    #vectorstore = Chroma.from_texts(texts=text_chunks , embedding=embeddings )
    #vectorstore2 = Chroma.from_texts(texts=text_chunks, embedding=embeddings, persist_directory="./chroma_db")
    return vectorstore
    
def get_conversation_chain(vectorstore):
    llm = ChatAnthropic(model='claude-3-sonnet-20240229')
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(k=10),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 ==0:
            st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="RAGE: Rade + RAG", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("I can help you chat about your documents.")
    user_question = st.text_input("Before we begin, please ensure you've uploaded your files from the sidebar.")
    if user_question:
        handle_userinput(user_question)


    with st.sidebar:
        st.subheader("Upload Your Files")
        pdf_docs = st.file_uploader("Upload Your Files and Press Process",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Please wait: it can take few minutes"):
                # get the pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                
                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
    
if __name__ == '__main__':
    main()