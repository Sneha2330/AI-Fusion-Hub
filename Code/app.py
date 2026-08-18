import gradio as gr
from audio_utils import transcribe_audio, summarize_text
from rag_utils import ensure_index, refresh_doc_list, answer_question
from sql_agents import run_sqlite_agent, run_postgres_agent


with gr.Blocks(title="AI Multi-Tool") as demo:

    # ---------------------------------------------------
    # TAB 1: AUDIO
    # ---------------------------------------------------
    with gr.Tab("Audio Transcription"):
        audio = gr.Audio(type="filepath", label="Upload or Record Audio")
        transcript = gr.Textbox(label="Transcript", lines=6)
        summary = gr.Textbox(label="Summary", lines=6)

        with gr.Row():
            trans_btn = gr.Button("Transcribe")
            sum_btn = gr.Button("Summarize Transcript")

        trans_btn.click(fn=transcribe_audio, inputs=audio, outputs=transcript)
        sum_btn.click(fn=summarize_text, inputs=transcript, outputs=summary)

    # ---------------------------------------------------
    # TAB 2: RAG CHATBOT
    # ---------------------------------------------------
    with gr.Tab("RAG Chatbot"):
        gr.Markdown(
            "Upload **PDF/DOCX/TXT**, click **Index Documents**, then ask questions.\n"
            "Use the dropdown to query a specific document or all documents."
        )

        with gr.Row():
            files = gr.Files(
                label="Upload files",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".txt"]
            )

        with gr.Row():
            index_btn = gr.Button("Index Documents", variant="primary")
            refresh_btn = gr.Button("Refresh list")

        with gr.Row():
            doc_dropdown = gr.Dropdown(
                choices=["All documents"],
                value="All documents",
                label="Query scope"
            )

        with gr.Row():
            question = gr.Textbox(
                label="Your question",
                lines=2,
                placeholder="Ask about documents"
            )

        with gr.Row():
            ask_btn = gr.Button("Ask")

        rag_answer = gr.Markdown(label="Answer")

        # ---- RAG wiring ----
        def _do_index(files_):
            msg = ensure_index(files_)
            return msg, gr.update(
                choices=refresh_doc_list(),
                value="All documents"
            )

        index_btn.click(fn=_do_index, inputs=files, outputs=[rag_answer, doc_dropdown])
        refresh_btn.click(
            fn=lambda: gr.update(choices=refresh_doc_list(), value="All documents"),
            inputs=None, outputs=doc_dropdown
        )
        ask_btn.click(fn=answer_question, inputs=[question, doc_dropdown], outputs=rag_answer)

    # ---------------------------------------------------
    # TAB 3: SQL AGENTS
    # ---------------------------------------------------
    with gr.Tab("SQL Agents"):

        gr.Markdown(
            "**SQLite Agent** executes generated SQL on a local SQLite DB.\n\n"
            "**Postgres Agent** generates SQL (no execution)."
        )

        sql_input = gr.Textbox(
            label="Your Query",
            placeholder=(
                "Examples:\n"
                "- Create a table employees(id integer primary key, name text);\n"
                "- Insert two rows: Alice, Bob.\n"
                "- Select all employees.\n"
            ),
            lines=3
        )

        with gr.Row():
            sqlite_btn = gr.Button("Run SQL Agent (SQLite)", variant="primary")
            postgres_btn = gr.Button("Generate Postgres SQL")

        with gr.Row():
            sqlite_out = gr.Markdown(label="SQLite Result")
            postgres_out = gr.Markdown(label="Postgres SQL")

        sqlite_btn.click(fn=run_sqlite_agent, inputs=sql_input, outputs=sqlite_out)
        postgres_btn.click(fn=run_postgres_agent, inputs=sql_input, outputs=postgres_out)


demo.launch(debug=True)
