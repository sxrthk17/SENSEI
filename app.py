import streamlit as st

from aesthetics import (
    initialize_theme,
    apply_theme,
    display_quote,
)

from memory_db import (
    init_db,
    log_interaction,
    get_all_history,
    get_total_questions_asked,
)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Sensei",
    page_icon="🥋",
    layout="wide",
)


# ============================================================
# Initialize Sensei
# ============================================================

theme = initialize_theme()
apply_theme(theme)

init_db()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.markdown("## 🥋 SENSEI")
    st.markdown("---")

    mode = st.radio(
        "Choose your mode",
        [
            "📝 Notes Q&A",
            "🧮 Math Solver",
            "📚 Study History",
        ],
    )

    st.markdown("---")

    st.metric(
        "Questions Asked",
        get_total_questions_asked(),
    )

    st.markdown("---")

    st.caption(
        "Your personal AI study assistant."
    )


# ============================================================
# Main Header
# ============================================================

st.markdown(
    """
    <div style="text-align: center;">
        <h1 class="sensei-accent">🥋 SENSEI</h1>
        <p>Your personal AI study assistant</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NOTES Q&A
# ============================================================

if mode == "📝 Notes Q&A":

    st.markdown("## 📝 Notes Q&A")

    st.info(
        "Upload your study notes and ask questions grounded in them."
    )

    display_quote()

    # --------------------------------------------------------
    # PDF Upload
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload your study notes",
        type=["pdf"],
    )

    if uploaded_file is not None:

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        # ----------------------------------------------------
        # Process Notes
        # ----------------------------------------------------

        if st.button("🧠 Process Notes"):

            from rag_pipeline import (
                create_embeddings,
                ingest_pdf,
            )

            with st.spinner(
                "Sensei is studying your notes..."
            ):

                # Save uploaded PDF
                with open(
                    uploaded_file.name,
                    "wb",
                ) as file:

                    file.write(
                        uploaded_file.getbuffer()
                    )

                # Create embedding model
                embeddings = create_embeddings()

                # Ingest PDF into ChromaDB
                vector_store = ingest_pdf(
                    uploaded_file.name,
                    embeddings,
                )

                # IMPORTANT:
                # Keep this exact vector store in session.
                st.session_state.vector_store = vector_store

                st.session_state.source_pdf = (
                    uploaded_file.name
                )

                # Mark notes as processed
                st.session_state.notes_processed = True

            st.success(
                "Notes processed successfully! 🥋"
            )


    # ========================================================
    # ASK SENSEI
    # ========================================================

    if "vector_store" in st.session_state:

        st.markdown("---")

        st.markdown("### 🎯 Ask Sensei")

        question = st.text_input(
            "Ask a question from your notes:",
            placeholder="e.g. What is a multistage graph?",
            key="sensei_question",
        )

        if st.button("🥋 Ask Sensei"):

            if not question.strip():

                st.warning(
                    "Please enter a question first."
                )

            else:

                from rag_pipeline import (
                    retrieve_documents,
                    generate_answer,
                )

                with st.spinner(
                    "Sensei is thinking..."
                ):

                    # ====================================================
                    # IMPORTANT:
                    # Use the SAME vector store that processed the PDF.
                    # Do NOT create another Chroma connection here.
                    # ====================================================

                    vector_store = (
                        st.session_state.vector_store
                    )

                    # Retrieve relevant chunks
                    documents = retrieve_documents(
                    vector_store,
                    question,
                    k=6,
                )

                    # Save retrieved documents temporarily
                    st.session_state.retrieved_documents = (
                        documents
                    )

                    # Generate grounded answer
                    answer = generate_answer(
                        question,
                        documents,
                    )

                    # Save interaction
                    log_interaction(
                        question=question,
                        answer=answer,
                        source_pdf=st.session_state.get(
                            "source_pdf",
                            "unknown",
                        ),
                    )

                # ====================================================
                # ANSWER
                # ====================================================

                st.markdown(
                    "### 🥋 Sensei's Answer"
                )

                st.markdown(
                    f"""
                    <div class="sensei-card">
                        {answer}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ====================================================
                # RETRIEVED NOTES
                # ====================================================

                with st.expander(
                    "🔎 View retrieved notes",
                    expanded=True,
                ):

                    if not documents:

                        st.error(
                            "No relevant chunks were retrieved from ChromaDB."
                        )

                    else:

                        st.success(
                            f"Retrieved {len(documents)} relevant chunks."
                        )

                        for index, document in enumerate(
                            documents,
                            start=1,
                        ):

                            st.markdown(
                                f"### Chunk {index}"
                            )

                            # Page information if available
                            metadata = document.metadata

                            if metadata:

                                source = metadata.get(
                                    "source",
                                    "Unknown",
                                )

                                page = metadata.get(
                                    "page",
                                    None,
                                )

                                if page is not None:

                                    st.caption(
                                        f"Source: {source} | "
                                        f"Page: {page + 1}"
                                    )

                                else:

                                    st.caption(
                                        f"Source: {source}"
                                    )

                            st.write(
                                document.page_content
                            )

                            st.markdown("---")


# ============================================================
# MATH SOLVER
# ============================================================

elif mode == "🧮 Math Solver":

    st.markdown("## 🧮 Math Solver")

    st.info(
        "Solve mathematical expressions and equations with Sensei."
    )

    display_quote()

    st.markdown(
        "### 🧮 Math Solver"
    )

    st.write(
        "Math Solver integration is coming next."
    )


# ============================================================
# STUDY HISTORY
# ============================================================

elif mode == "📚 Study History":

    st.markdown("## 📚 Study History")

    st.info(
        "Review your previous questions and answers."
    )

    display_quote()

    history = get_all_history()

    if not history:

        st.info(
            "No study history yet. Ask Sensei a question!"
        )

    else:

        st.markdown(
            f"### 📚 {len(history)} Questions"
        )

        for timestamp, question, answer in history:

            with st.expander(
                f"❓ {question}"
            ):

                st.caption(timestamp)

                st.markdown(
                    "**Answer:**"
                )

                st.write(answer)