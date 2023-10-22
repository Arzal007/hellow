import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
import fitz  # PyMuPDF for PDF extraction

st.set_page_config(page_title="Chat with the Bain Report", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Chat with the Bain Report")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Bain Report!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Bain Report – hang tight! This should take 1-2 minutes."):
        # Replace this path with your actual data directory containing PDF files.
        data_dir = "./data"
        
        # Initialize an empty list to store Document objects.
        docs = []

        # Iterate over PDF files in the data directory and extract text.
        for pdf_file in os.listdir(data_dir):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(data_dir, pdf_file)
                pdf_document = fitz.open(pdf_path)
                text = ""
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    text += page.get_text()

                document = Document(text, title=pdf_file, source=pdf_path)
                docs.append(document)

        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Bain Report and your job is to answer technical questions. Assume that all questions are related to the Bain Report. Keep your answers technical and based on facts – do not hallucinate features."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If the last message is not from the assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add the response to the message history
