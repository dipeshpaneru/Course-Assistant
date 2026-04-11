import sys
import gradio as gr
from models import get_fine_tuned_model
from rag import RAG
from rag_inference import RAGInference


model, tokenizer = get_fine_tuned_model()
rag = RAG()
rag_model = RAGInference(model, tokenizer, rag)

def upload_pdf(file):
    if file is None:
        return "No file uploaded."
    rag.add_pdf(file.name)
    return f"File added to knowledge base, Total docs in store: {rag.collection.count()}"


def chat(question, history):
    if not question:
        return "", history
    answer = rag_model.generate(question)
    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return "", history


with gr.Blocks(title="AI Course Assistant") as demo:

    with gr.Tab("Chat"):
        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Ask a question...")
        clear = gr.Button("Clear")
        msg.submit(chat, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: [], outputs=chatbot)

    with gr.Tab("Upload PDF"):
        pdf_input = gr.File(label="Upload lecture PDF", file_types=[".pdf"])
        status = gr.Textbox(label="Status", show_label=True)
        upload_btn = gr.Button("Add to Knowledge Base")
        upload_btn.click(upload_pdf, inputs=pdf_input, outputs=status)

demo.launch()

