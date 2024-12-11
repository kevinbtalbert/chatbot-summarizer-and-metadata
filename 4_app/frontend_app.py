import os
import gradio
from typing import Any, List, Tuple
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import chromadb
import openai
from sklearn.feature_extraction.text import CountVectorizer

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_KEY")

# Initialize Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="/home/cdsw/chroma_db")

# Create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize LangChain Chroma vector store
langchain_chroma = Chroma(
    client=chroma_client,
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embedding_function,
)

# Define Gradio CSS styles
app_css = """
    .gradio-header {
        color: white;
    }
    .gradio-description {
        color: white;
    }
    #custom-logo {
        text-align: center;
    }
    .gr-interface {
        background-color: rgba(255, 255, 255, 0.8);
    }
    .gradio-header {
        background-color: rgba(0, 0, 0, 0.5);
    }
    .gradio-input-box, .gradio-output-box {
        background-color: rgba(255, 255, 255, 0.8);
    }
    h1 {
        color: white; 
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: large; !important;
    }
"""

# Helper function to extract top keywords
def extract_keywords(text: str, num_keywords: int = 5) -> List[str]:
    vectorizer = CountVectorizer(max_features=num_keywords, stop_words='english')
    vectorizer.fit([text])
    keywords = vectorizer.get_feature_names_out()
    return list(keywords)

# Helper function for generating responses
def get_responses(num_docs: int, summary_display: str, full_doc_display: str, question: str) -> Tuple[str, str, str, List[str]]:
    if not question or not num_docs:
        return "One or more fields have not been specified.", "", "", []

    if not full_doc_display:
        full_doc_display = "No"

    source, doc_snippet = query_chroma_vectordb(question, full_doc_display, num_docs)
    if summary_display == "Yes":
        summary = get_llm_response_with_context(question, doc_snippet, "gpt-4")
    else:
        summary = "Summary disabled for this search."

    keywords = extract_keywords(doc_snippet)
    
    return summary, source, doc_snippet, keywords

# Query the Chroma vector database
def query_chroma_vectordb(query: str, full_doc_display: str, num_docs: int) -> Tuple[str, str]:
    docs = langchain_chroma.similarity_search(query, num_docs)
    doc_snippet = []
    source_info = []

    for i, doc in enumerate(docs):
        if full_doc_display == "Yes":
            doc_snippet.append(f"Doc {i + 1}: Relevant content: {doc.page_content}")
        source_info.append(f"Doc {i + 1}: Source link: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")

    doc_snippet_str = "\n".join(doc_snippet) if full_doc_display == "Yes" else "Show document response turned off."
    source_info_str = "\n".join(source_info)

    return source_info_str, doc_snippet_str

# Generate response using OpenAI API
def get_llm_response_with_context(question: str, context: str, engine: str) -> str:
    question = f"Summarize the content provided in a paragraph or less."

    response = openai.ChatCompletion.create(
        model=engine,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": question},
        ]
    )
    return response['choices'][0]['message']['content']

# Main function to configure and launch Gradio app
def main():
    print("Configuring Gradio app")

    DESC = "This app leverages Chroma's vector database for semantic search and integrates Generative AI for summarization."
    demo = gradio.Interface(
        fn=get_responses,
        title="Air Force Seek Eagle Office AI-Powered Search",
        description=DESC,
        inputs=[
            gradio.Slider(minimum=1, maximum=10, step=1, value=3, label="Select number of similar documents to return"),
            gradio.Radio(["Yes", "No"], label="Show summary (Generative AI)", value="Yes"),
            gradio.Radio(["Yes", "No"], label="Show full document extract", value="Yes"),
            gradio.Textbox(label="Question / Keywords", placeholder="Enter your search here"),
        ],
        outputs=[
            gradio.Textbox(label="LLM Summary"),
            gradio.Textbox(label="Data Source(s) and Page Reference"),
            gradio.Textbox(label="Document Response"),
            gradio.Textbox(label="Top Keywords"),
        ],
        allow_flagging="never",
        css=app_css,
    )

    print("Launching Gradio app")
    demo.launch(
        share=True,
        show_error=True,
        server_name="127.0.0.1",
        server_port=int(os.getenv("CDSW_APP_PORT", 8100)),
    )
    print("Gradio app ready")

if __name__ == "__main__":
    main()
