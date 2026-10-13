
import ast
import html
import random
import streamlit as st
import time
from streamlit_autorefresh import st_autorefresh

from aesthetics import (
    initialize_theme,
    apply_theme,
    display_quote,
    display_mascot,
    render_theme_selector,
)

from memory_db import (
    init_db,
    init_learning_db,
    log_interaction,
    get_all_history,
    get_total_questions_asked,
    save_pomodoro_session,
    get_pomodoro_stats,
    get_recent_pomodoro_sessions,
    add_todo_task,
    get_todo_tasks,
    set_todo_completed,
    delete_todo_task,
)


from question_engine import (
    extract_concepts,
    generate_questions_for_concept,
    grade_question,
    calculate_quiz_score,
    next_difficulty,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sensei",
    page_icon="🥋",
    layout="wide",
)


# ============================================================
# INITIALIZE SENSEI
# ============================================================

theme = initialize_theme()
apply_theme(theme)
init_db()
init_learning_db()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🥋 SENSEI")
    st.caption("Your personal AI study workspace.")

    if "sidebar_mascot_stage" not in st.session_state:
        st.session_state.sidebar_mascot_stage = random.randint(1, 5)

    display_mascot(stage=st.session_state.sidebar_mascot_stage)
    st.divider()

    mode = st.radio(
    "Choose your mode",
    [
        "📝 Notes Q&A",
        "🧠 Adaptive Quiz",
        "🧮 Math Solver",
        "📚 Study History",
        "🍅 Pomodoro Focus",
        "✨ To-do List",
    ],
    key="sensei_navigation",
)

    st.divider()
    render_theme_selector()
    st.divider()

    st.metric(
        "Questions Asked",
        get_total_questions_asked(),
    )
    
    st.divider()

    st.caption("Learn. Practice. Improve.")


# ============================================================
# MAIN HEADER — SENSEI HERO
# ============================================================

hero_left, hero_right = st.columns([1.65, 1], gap="large")

with hero_left:
    st.markdown(
        """
        <div class="sensei-hero">
            <div class="sensei-eyebrow">
                YOUR PERSONAL AI LEARNING SPACE
            </div>
            <h1>Welcome to <span class="sensei-accent">SENSEI</span></h1>
            <p class="sensei-subtitle">
                Learn with focus. Practice with purpose.
                Turn knowledge into mastery.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sensei-card">
            <div class="sensei-eyebrow">YOUR DOJO</div>
            <strong>One workspace. Three ways to grow.</strong>
            <p class="sensei-muted">
                Explore your notes, solve mathematical problems,
                and revisit your learning journey.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    display_mascot(stage=6)

st.markdown(
    '<hr class="sensei-divider">',
    unsafe_allow_html=True,
)


# ============================================================
# NOTES Q&A
# ============================================================

# ============================================================
# NOTES Q&A — MULTI-PDF LIBRARY (SOURCE-FIRST)
# Replace your existing NOTES Q&A section with this block.
# ============================================================

if mode == "📝 Notes Q&A":

    import hashlib
    import json
    from pathlib import Path

    import streamlit as st

    from rag_pipeline import (
        create_embeddings,
        ingest_pdf,
        generate_document_id,
        load_existing_vector_store,
        retrieve_documents,
        generate_answer,
        is_syllabus_query

    )

    st.markdown("## 📝 Notes Q&A")
    st.info(
        "Upload multiple study PDFs, organize them by subject, "
        "and choose exactly which notes Sensei should use."
    )

    display_quote()

    # --------------------------------------------------------
    # PERSISTENT LIBRARY
    # --------------------------------------------------------

    upload_dir = Path("uploaded_notes")
    upload_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = upload_dir / "library.json"

    if "sensei_library" not in st.session_state:
        library = {}

        if manifest_path.exists():
            try:
                loaded = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                if isinstance(loaded, dict):
                    library = loaded
            except (json.JSONDecodeError, OSError):
                st.warning(
                    "The library manifest could not be read. "
                    "Recovering entries from saved PDFs where possible."
                )

        # Recover PDFs saved by the earlier uploader.
        # This does not delete or reset ChromaDB.
        known_paths = {
            str(Path(item["path"]).resolve())
            for item in library.values()
            if isinstance(item, dict) and item.get("path")
        }

        for pdf_path in upload_dir.glob("*.pdf"):
            if str(pdf_path.resolve()) in known_paths:
                continue

            try:
                document_id = generate_document_id(str(pdf_path))
            except Exception:
                continue

            display_name = pdf_path.name

            # Earlier uploads used:
            # <12-character-hash>_<original-filename>.pdf
            if (
                len(display_name) > 13
                and display_name[12] == "_"
            ):
                display_name = display_name[13:]

            library.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "display_name": display_name,
                    "path": str(pdf_path),
                    "subject": "Uncategorized",
                },
            )

        st.session_state.sensei_library = library

        manifest_path.write_text(
            json.dumps(library, indent=2),
            encoding="utf-8",
        )

    library = st.session_state.sensei_library

    # --------------------------------------------------------
    # UPLOAD AND PROCESS MULTIPLE PDFs
    # --------------------------------------------------------

    st.markdown("### 📥 Add study notes")

    subject_name = st.text_input(
        "Subject / course for these PDFs",
        value="Uncategorized",
        placeholder="e.g. Applications of AI",
        key="upload_subject_name",
    )

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="notes_pdf_uploader",
    )

    if uploaded_files:
        st.caption(
            f"{len(uploaded_files)} PDF file(s) selected."
        )

    if st.button(
        "🧠 Process selected PDFs",
        key="process_notes_button",
        disabled=not uploaded_files,
        use_container_width=True,
    ):
        new_document_ids = []
        errors = []

        try:
            with st.spinner(
                "🥋 Sensei is processing your study library..."
            ):
                embeddings = create_embeddings()

                for uploaded_file in uploaded_files:
                    try:
                        file_bytes = uploaded_file.getvalue()
                        content_hash = hashlib.sha256(
                            file_bytes
                        ).hexdigest()
                        document_id = content_hash

                        # If the file is already in the manifest,
                        # reuse it without embedding it again.
                        if document_id in library:
                            existing = library[document_id]
                            existing_path = Path(
                                existing.get("path", "")
                            )

                            if existing_path.exists():
                                new_document_ids.append(document_id)
                                continue

                        safe_name = Path(
                            uploaded_file.name
                        ).name

                        pdf_path = upload_dir / (
                            f"{content_hash[:12]}_{safe_name}"
                        )

                        pdf_path.write_bytes(file_bytes)

                        ingest_pdf(
                            str(pdf_path),
                            embeddings,
                            subject=(
                                subject_name.strip()
                                or "Uncategorized"
                            ),
                        )

                        library[document_id] = {
                            "document_id": document_id,
                            "display_name": uploaded_file.name,
                            "path": str(pdf_path),
                            "subject": (
                                subject_name.strip()
                                or "Uncategorized"
                            ),
                        }

                        new_document_ids.append(document_id)

                    except Exception as error:
                        errors.append(
                            f"{uploaded_file.name}: {error}"
                        )

                manifest_path.write_text(
                    json.dumps(library, indent=2),
                    encoding="utf-8",
                )

                st.session_state.sensei_library = library

                if new_document_ids:
                    st.session_state[
                        "document_scope_selector"
                    ] = list(dict.fromkeys(new_document_ids))

                st.session_state.vector_store = (
                    load_existing_vector_store(embeddings)
                )

            if new_document_ids:
                st.success(
                    f"Library ready: {len(library)} PDF(s) available."
                )

            for error in errors:
                st.error(f"Could not process {error}")

        except Exception as error:
            st.error(
                f"Could not initialize the document library: {error}"
            )

    # --------------------------------------------------------
    # LIBRARY VIEW AND DOCUMENT SELECTION
    # --------------------------------------------------------

    library = st.session_state.sensei_library

    st.divider()
    st.markdown("### 📚 Your study library")
    st.caption(
        "Your library manifest and uploaded PDF files persist "
        "between Streamlit reruns."
    )

    if not library:
        st.info("Upload and process your first PDF to get started.")

    else:
        st.metric("PDFs in library", len(library))

        for document in library.values():
            st.markdown(
                f"- **{document.get('display_name', 'Unnamed PDF')}** "
                f"— {document.get('subject', 'Uncategorized')}"
            )

        document_ids = list(library.keys())

        saved_selection = st.session_state.get(
            "document_scope_selector",
            [document_ids[-1]],
        )

        saved_selection = [
            doc_id
            for doc_id in saved_selection
            if doc_id in library
        ]

        if not saved_selection:
            saved_selection = [document_ids[-1]]

        st.session_state["document_scope_selector"] = (
            saved_selection
        )

        selected_document_ids = st.multiselect(
            "Search within these documents",
            options=document_ids,
            format_func=lambda doc_id: (
                f"{library[doc_id].get('display_name', 'Unnamed PDF')} "
                f"· {library[doc_id].get('subject', 'Uncategorized')}"
            ),
            key="document_scope_selector",
        )
        # Save selected PDFs so Adaptive Quiz can access them
        st.session_state["selected_document_ids"] = list(selected_document_ids)

        # ----------------------------------------------------
        # ASK SENSEI — SOURCE-FIRST RETRIEVAL
        # ----------------------------------------------------

        st.divider()
        st.markdown("### 🎯 Ask Sensei")

        question = st.text_input(
            "Ask a question from your selected notes:",
            placeholder=(
                "e.g. What topics are covered in my AAI syllabus?"
            ),
            key="sensei_question",
        )

        # External research is disabled by default.
        # Enable it deliberately for questions needing outside context.
        use_web_research = st.checkbox(
            "Allow external web research if my notes are insufficient",
            value=False,
            key="allow_external_research",
            help=(
                "When unchecked, answers must use selected notes. "
                "Syllabus questions always remain notes-only."
            ),
        )

        if st.button(
            "🥋 Ask Sensei",
            key="ask_sensei_button",
            use_container_width=True,
        ):
            if not question.strip():
                st.warning("Please enter a question first.")

            elif not selected_document_ids:
                st.warning(
                    "Select at least one PDF from your library."
                )

            else:
                try:
                    with st.spinner(
                        "🥋 Searching your selected notes..."
                    ):
                        embeddings = create_embeddings()

                        vector_store = st.session_state.get(
                            "vector_store"
                        )

                        if vector_store is None:
                            vector_store = load_existing_vector_store(
                                embeddings
                            )
                            st.session_state.vector_store = (
                                vector_store
                            )

                        documents = retrieve_documents(
                            vector_store,
                            question,
                            k=6,
                            document_ids=selected_document_ids,
                        )

                        question_lower = question.casefold()

                        # Treat syllabus requests as exact-source
                        # extraction, not general exam research.
                        # Detect syllabus questions using the shared helper
                        syllabus_question = is_syllabus_query(question)

                        selected_names = [
                            library[doc_id].get("display_name", "Unnamed PDF")
                            for doc_id in selected_document_ids
                        ]

                        # External research is allowed only when:
                        # 1. The user enabled it.
                        # 2. The question is not about a syllabus.
                        # 3. No relevant content was retrieved from the selected PDFs.
                        allow_web = (
                            use_web_research
                            and not syllabus_question
                        )

                        if documents:
                            answer = generate_answer(
                                question,
                                documents,
                                use_web=allow_web,
                            )

                        elif allow_web:
                            # No notes were retrieved, and the user allowed web research.
                            answer = generate_answer(
                                question,
                                [],
                                use_web=True,
                            )

                        else:
                            answer = (
                                "I couldn't find relevant content in "
                                "the selected PDFs. No external research "
                                "was performed.\n\n"
                                "Try selecting the correct syllabus PDF, "
                                "checking the retrieved notes, or "
                                "reprocessing that document."
                            )
                        log_interaction(
                            question=question,
                            answer=answer,
                            source_pdf=", ".join(selected_names),
                        )

                        st.session_state.last_answer = answer
                        st.session_state.last_question = question
                        st.session_state.last_documents = documents
                        st.session_state.last_scope_names = selected_names

                except Exception as error:
                    st.error(
                        f"Could not answer the question: {error}"
                    )

        # ----------------------------------------------------
        # ANSWER AND RETRIEVED EVIDENCE
        # ----------------------------------------------------

        if st.session_state.get("last_answer"):
            st.markdown("### 🥋 Sensei's Answer")
            st.markdown(st.session_state.last_answer)

            with st.expander(
                "📖 View retrieved notes",
                expanded=False,
            ):
                documents = st.session_state.get(
                    "last_documents", []
                )

                if not documents:
                    st.info(
                        "No note chunks were retrieved for the last "
                        "question. The answer above should not be "
                        "treated as a summary of your PDFs."
                    )

                else:
                    st.success(
                        f"Retrieved {len(documents)} chunk(s) "
                        "from the selected library scope."
                    )

                    for index, document in enumerate(
                        documents,
                        start=1,
                    ):
                        metadata = document.metadata
                        doc_id = metadata.get("document_id")

                        friendly_name = (
                            library.get(doc_id, {}).get("display_name")
                            or metadata.get("source_pdf", "Unknown PDF")
                        )

                        page = metadata.get("page_number")
                        chunk_index = metadata.get("chunk_index")

                        st.markdown(f"#### Chunk {index}")
                        st.caption(
                            f"📄 {friendly_name}"
                            + (
                                f" · Page {page}"
                                if page is not None
                                else ""
                            )
                        )

                        if chunk_index is not None:
                            st.caption(f"Chunk index: {chunk_index}")

                        st.write(document.page_content)
                        st.divider()

## QUIZ GENERATOR !!


elif mode == "🧠 Adaptive Quiz":
    import json
    import random
    from rag_pipeline import (
        create_embeddings,
        load_existing_vector_store,
        retrieve_documents,
    )
    from question_engine import (
        QuestionEngineError,
        extract_concepts,
        generate_questions_for_concept,
        grade_question,
        calculate_quiz_score,
        next_difficulty,
    )

    st.title("🧠 SENSEI Adaptive Quiz")
    st.caption("Test your understanding using evidence from your study PDFs.")

    # ---------------------------------------------------------
    # Session state
    # ---------------------------------------------------------
    defaults = {
        "sensei_quiz_questions": [],
        "sensei_quiz_results": [],
        "sensei_quiz_submitted": False,
        "sensei_quiz_scores": [],
        "sensei_quiz_difficulty": 3,
        "sensei_quiz_source_names": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # ---------------------------------------------------------
    # Quiz settings
    # ---------------------------------------------------------
    st.subheader("Configure your quiz")

    quiz_count = st.selectbox(
        "Number of questions",
        [3, 5, 10],
        index=0,
        key="sensei_quiz_count",
    )

    difficulty = st.slider(
        "Target difficulty",
        min_value=1,
        max_value=5,
        value=st.session_state.sensei_quiz_difficulty,
        help="1 = easier questions, 5 = more challenging questions.",
        key="sensei_quiz_difficulty_slider",
    )

    st.caption(
        f"Current target difficulty: {difficulty}/5"
    )

    # ---------------------------------------------------------
    # Generate a PDF-grounded quiz
    # ---------------------------------------------------------
    if st.button(
        "🎯 Generate PDF-Based Quiz",
        type="primary",
        key="sensei_generate_quiz",
    ):
        if not st.session_state.get("selected_document_ids"):
            st.warning(
                "Select at least one PDF in the Notes Q&A section first."
            )
        else:
            try:
                with st.spinner(
                    "Reading your selected PDFs and verifying question evidence..."
                ):
                    embeddings = create_embeddings()
                    vector_store = load_existing_vector_store(embeddings)

                    if vector_store is None:
                        raise QuestionEngineError(
                            "The existing vector store could not be loaded. "
                            "Please ingest your PDFs in Notes Q&A first."
                        )

                    selected_ids = st.session_state.get("selected_document_ids", [])
                    st.write("PDFs available to quiz:", selected_ids)

                    retrieved_docs = retrieve_documents(
                        vector_store,
                        (
                            "Important definitions, principles, explanations, "
                            "algorithms, examples, and concepts in the selected "
                            "study documents"
                        ),
                        k=12,
                        document_ids=selected_ids,
                    )

                    if not retrieved_docs:
                        raise QuestionEngineError(
                            "No relevant chunks were retrieved for the selected PDFs."
                        )

                    concepts = extract_concepts(
                        retrieved_docs,
                        max_concepts=8,
                    )

                    candidates = []
                    seen = set()

                    for concept in concepts:
                        # Restrict a concept to its identified source pages.
                        concept_pages = set(concept.get("pages", []))

                        if concept_pages:
                            concept_docs = [
                                doc
                                for doc in retrieved_docs
                                if str(
                                    (getattr(doc, "metadata", {}) or {}).get(
                                        "page_number",
                                        (getattr(doc, "metadata", {}) or {}).get(
                                            "page", "unknown"
                                        ),
                                    )
                                ) in concept_pages
                            ]
                        else:
                            concept_docs = retrieved_docs

                        # Do not silently use unrelated pages if page matching fails.
                        if not concept_docs:
                            continue

                        generated = generate_questions_for_concept(
                            documents=concept_docs,
                            concept_name=concept["name"],
                            questions_per_concept=min(3, quiz_count),
                        )

                        for question in generated:
                            if question["question"].casefold() in seen:
                                continue

                            seen.add(question["question"].casefold())
                            candidates.append(question)

                    # Prefer the requested difficulty, then the nearest available.
                    candidates.sort(
                        key=lambda q: abs(
                            int(q.get("difficulty", 1)) - difficulty
                        )
                    )

                    candidates = candidates[:quiz_count]
                    random.shuffle(candidates)

                    if not candidates:
                        raise QuestionEngineError(
                            "No questions passed evidence verification. "
                            "Try another PDF or a different selection."
                        )

                    st.session_state.sensei_quiz_questions = candidates
                    st.session_state.sensei_quiz_results = []
                    st.session_state.sensei_quiz_submitted = False
                    st.session_state.sensei_quiz_source_names = sorted({
                        q.get("source_pdf", "Unknown PDF")
                        for q in candidates
                    })

                st.success(
                    f"Generated {len(candidates)} evidence-verified questions."
                )
                st.rerun()

            except Exception as exc:
                st.error(f"Could not generate the quiz: {exc}")

    # ---------------------------------------------------------
    # Display quiz and collect answers
    # ---------------------------------------------------------
    questions = st.session_state.sensei_quiz_questions

    if questions:
        st.divider()
        st.subheader("Your quiz")

        st.write(
            f"**Questions:** {len(questions)}  |  "
            f"**Target difficulty:** {difficulty}/5"
        )

        with st.expander("PDFs used for this quiz"):
            for name in st.session_state.sensei_quiz_source_names:
                st.write(f"📄 {name}")

        if not st.session_state.sensei_quiz_submitted:
            with st.form("sensei_adaptive_quiz_form"):
                submitted_answers = {}

                for index, question in enumerate(questions):
                    st.markdown(
                        f"**Q{index + 1}. {question['question']}**"
                    )

                    question_type = question["question_type"]
                    answer_key = json.loads(question["answer_key"])

                    if question_type == "multiple_choice":
                        options = answer_key.get("options", {})

                        selected_label = st.radio(
                            "Choose one answer",
                            options=list(options.keys()),
                            format_func=lambda label, opts=options: (
                                f"{label}. {opts[label]}"
                            ),
                            key=f"sensei_quiz_answer_{index}",
                        )

                        submitted_answers[index] = selected_label

                    elif question_type == "true_false":
                        submitted_answers[index] = st.radio(
                            "Choose your answer",
                            ["true", "false"],
                            format_func=lambda value: value.title(),
                            key=f"sensei_quiz_answer_{index}",
                        )

                    else:
                        submitted_answers[index] = st.text_input(
                            "Your answer",
                            key=f"sensei_quiz_answer_{index}",
                            placeholder="Type your answer...",
                        )

                    st.caption(
                        f"Difficulty {question.get('difficulty', 1)}/5"
                    )
                    st.divider()

                submit_quiz = st.form_submit_button(
                    "✅ Submit Quiz",
                    type="primary",
                )

            if submit_quiz:
                results = []

                for index, question in enumerate(questions):
                    student_answer = submitted_answers.get(index, "")

                    # Empty responses are recorded as unanswered, not correct.
                    if not str(student_answer).strip():
                        results.append({
                            "is_correct": False,
                            "unanswered": True,
                            "question": question["question"],
                            "correct_answer": "",
                            "explanation": "",
                        })
                        continue

                    graded = grade_question(
                        question["question_type"],
                        question["answer_key"],
                        student_answer,
                    )

                    graded["question"] = question["question"]
                    graded["student_answer"] = student_answer
                    graded["unanswered"] = False
                    results.append(graded)

                st.session_state.sensei_quiz_results = results
                st.session_state.sensei_quiz_submitted = True

                score = calculate_quiz_score([
                    result for result in results
                    if not result.get("unanswered")
                ])

                st.session_state.sensei_quiz_scores.append(
                    score["percentage"]
                )
                st.session_state.sensei_quiz_scores = (
                    st.session_state.sensei_quiz_scores[-5:]
                )

                st.rerun()

        # -----------------------------------------------------
        # Results and evidence review
        # -----------------------------------------------------
        if st.session_state.sensei_quiz_submitted:
            results = st.session_state.sensei_quiz_results
            answered = [
                result for result in results
                if not result.get("unanswered")
            ]
            score = calculate_quiz_score(answered)

            st.divider()
            st.subheader("📊 Your results")

            col1, col2, col3 = st.columns(3)

            col1.metric("Score", f"{score['percentage']:.1f}%")
            col2.metric("Correct", score["correct"])
            col3.metric(
                "Unanswered",
                sum(r.get("unanswered", False) for r in results),
            )

            for index, result in enumerate(results):
                if result.get("unanswered"):
                    heading = "⏭️ Unanswered"
                elif result.get("is_correct"):
                    heading = "✅ Correct"
                else:
                    heading = "❌ Incorrect"

                with st.expander(
                    f"{heading} — Question {index + 1}",
                    expanded=not result.get("is_correct", False),
                ):
                    st.write(result.get("question", ""))

                    if result.get("unanswered"):
                        st.info("You did not answer this question.")
                    else:
                        st.write(
                            f"**Your answer:** "
                            f"{result.get('student_answer', '')}"
                        )

                    st.write(
                        f"**Expected answer:** "
                        f"{result.get('correct_answer', '')}"
                    )

                    if result.get("explanation"):
                        st.write(
                            f"**Explanation:** {result['explanation']}"
                        )

                    if result.get("evidence_quote"):
                        st.markdown("**Verified PDF evidence**")
                        st.info(result["evidence_quote"])

                        st.caption(
                            f"Source: {result.get('source_pdf', 'Unknown PDF')} "
                            f"| Page {result.get('page_number', 'Unknown')}"
                        )

            # Adapt difficulty based on recent quiz performance.
            if len(st.session_state.sensei_quiz_scores) >= 2:
                recommended = next_difficulty(
                    st.session_state.sensei_quiz_scores,
                    current_difficulty=difficulty,
                )

                st.divider()
                st.write(
                    f"**Recommended next difficulty:** {recommended}/5"
                )

                if recommended > difficulty:
                    st.success(
                        "Great progress! You may be ready for harder questions."
                    )
                elif recommended < difficulty:
                    st.info(
                        "A little more practice at an easier level may help."
                    )
                else:
                    st.info(
                        "Keep practising at this level to build consistency."
                    )

                if st.button(
                    "Use Recommended Difficulty",
                    key="sensei_use_recommended_difficulty",
                ):
                    st.session_state.sensei_quiz_difficulty = recommended
                    st.rerun()

            if st.button(
                "🔄 Generate Another Quiz",
                key="sensei_another_quiz",
            ):
                st.session_state.sensei_quiz_questions = []
                st.session_state.sensei_quiz_results = []
                st.session_state.sensei_quiz_submitted = False
                st.rerun()

# ============================================================
# MATH SOLVER
# ============================================================

elif mode == "🧮 Math Solver":

    st.markdown("## 🧮 Math Solver")
    st.caption("Powered by Sensei's mathematical engine.")

    tab_numerical, tab_linear, tab_statistics = st.tabs(
        [
            "📐 Numerical Math & Calculus",
            "🔢 Linear Algebra",
            "📊 Probability & Statistics",
        ]
    )

    # --------------------------------------------------------
    # TAB 1 — NUMERICAL MATH & CALCULUS
    # --------------------------------------------------------

    with tab_numerical:

        operation = st.selectbox(
            "Choose operation",
            [
                "Evaluate Expression",
                "Simplify Expression",
                "Expand Expression",
                "Factor Expression",
                "Solve Equation",
                "Solve System of Equations",
                "Differentiate",
                "Integrate",
                "Definite Integral",
                "Limit",
                "Series Expansion",
            ],
            key="numerical_operation",
        )

        if operation == "Solve System of Equations":

            equations_text = st.text_area(
                "Enter equations (one per line)",
                placeholder="x + y = 10\n2*x - y = 3",
                key="system_equations",
            )

            if st.button("Solve System", key="solve_system_button"):

                from math_solver import solve_system

                equations = [
                    line.strip()
                    for line in equations_text.splitlines()
                    if line.strip()
                ]

                try:
                    result = solve_system(equations)

                    if result["success"]:
                        st.success("Solution found")

                        for solution in result["result"]:
                            for variable, value in solution.items():
                                st.markdown(f"**{variable} = {value}**")
                    else:
                        st.error(result["error"])

                except Exception as error:
                    st.error(f"Could not solve the system: {error}")

        elif operation in [
            "Evaluate Expression",
            "Simplify Expression",
            "Expand Expression",
            "Factor Expression",
            "Solve Equation",
        ]:

            expression = st.text_input(
                "Enter expression/equation",
                placeholder=(
                    "Example: x + 2 = 5"
                    if operation == "Solve Equation"
                    else "Example: (x + 2)^2"
                ),
                key="basic_expression",
            )

            if st.button("Calculate", key="basic_math_button"):

                from math_solver import (
                    evaluate_expression,
                    simplify_expression,
                    expand_expression,
                    factor_expression,
                    solve_equation,
                )

                functions = {
                    "Evaluate Expression": evaluate_expression,
                    "Simplify Expression": simplify_expression,
                    "Expand Expression": expand_expression,
                    "Factor Expression": factor_expression,
                    "Solve Equation": solve_equation,
                }

                try:
                    result = functions[operation](expression)

                    if result["success"]:
                        st.success(f"Result: {result['result']}")
                    else:
                        st.error(result["error"])

                except Exception as error:
                    st.error(f"Calculation failed: {error}")

        else:

            expression = st.text_input(
                "Enter expression",
                placeholder="Example: x^3",
                key="calculus_expression",
            )

            variable = st.text_input(
                "Variable",
                value="x",
                key="calculus_variable",
            )

            if operation == "Definite Integral":

                lower = st.text_input(
                    "Lower limit",
                    value="0",
                    key="integral_lower",
                )

                upper = st.text_input(
                    "Upper limit",
                    value="1",
                    key="integral_upper",
                )

            if operation == "Limit":

                point = st.text_input(
                    "Approach point",
                    value="0",
                    key="limit_point",
                )

                direction = st.selectbox(
                    "Direction",
                    ["both", "+", "-"],
                    key="limit_direction",
                )

            if operation == "Series Expansion":

                point = st.text_input(
                    "Expansion point",
                    value="0",
                    key="series_point",
                )

                order = st.number_input(
                    "Order",
                    min_value=1,
                    value=6,
                    step=1,
                    key="series_order",
                )

            if st.button("Calculate", key="calculus_button"):

                from math_solver import (
                    differentiate,
                    integrate,
                    definite_integral,
                    calculate_limit,
                    series_expansion,
                )

                try:
                    if operation == "Differentiate":
                        result = differentiate(expression, variable)

                    elif operation == "Integrate":
                        result = integrate(expression, variable)

                    elif operation == "Definite Integral":
                        result = definite_integral(
                            expression,
                            variable,
                            lower,
                            upper,
                        )

                    elif operation == "Limit":
                        result = calculate_limit(
                            expression,
                            variable,
                            point,
                            direction,
                        )

                    else:
                        result = series_expansion(
                            expression,
                            variable,
                            point,
                            int(order),
                        )

                    if result["success"]:
                        st.success(f"Result: {result['result']}")
                    else:
                        st.error(result["error"])

                except Exception as error:
                    st.error(f"Calculation failed: {error}")

    # --------------------------------------------------------
    # TAB 2 — LINEAR ALGEBRA
    # --------------------------------------------------------

    with tab_linear:

        from math_solver import (
            vector_add,
            vector_subtract,
            dot_product,
            vector_norm,
            matrix_add,
            matrix_multiply,
            matrix_transpose,
            matrix_determinant,
            matrix_inverse,
            matrix_rank,
            matrix_eigenvalues,
            matrix_eigenvectors,
        )

        operation = st.selectbox(
            "Choose operation",
            [
                "Vector Addition",
                "Vector Subtraction",
                "Dot Product",
                "Vector Norm",
                "Matrix Addition",
                "Matrix Multiplication",
                "Matrix Transpose",
                "Matrix Determinant",
                "Matrix Inverse",
                "Matrix Rank",
                "Matrix Eigenvalues",
                "Matrix Eigenvectors",
            ],
            key="linear_operation",
        )

        if operation in [
            "Vector Addition",
            "Vector Subtraction",
            "Dot Product",
        ]:

            vector_a = st.text_input(
                "Vector A",
                placeholder="[1, 2, 3]",
                key="vector_a",
            )

            vector_b = st.text_input(
                "Vector B",
                placeholder="[4, 5, 6]",
                key="vector_b",
            )

            if st.button("Calculate", key="vector_operation_button"):

                try:
                    a = ast.literal_eval(vector_a)
                    b = ast.literal_eval(vector_b)

                    if operation == "Vector Addition":
                        result = vector_add(a, b)
                    elif operation == "Vector Subtraction":
                        result = vector_subtract(a, b)
                    else:
                        result = dot_product(a, b)

                    if result["success"]:
                        st.success(f"Result: {result['result']}")
                    else:
                        st.error(result["error"])

                except Exception as error:
                    st.error(f"Invalid vector input: {error}")

        elif operation == "Vector Norm":

            vector_text = st.text_input(
                "Vector",
                placeholder="[3, 4]",
                key="vector_norm_input",
            )

            if st.button("Calculate", key="vector_norm_button"):

                try:
                    vector = ast.literal_eval(vector_text)
                    result = vector_norm(vector)

                    if result["success"]:
                        st.success(f"Result: {result['result']}")
                    else:
                        st.error(result["error"])

                except Exception as error:
                    st.error(f"Invalid vector input: {error}")

        else:

            matrix_a = st.text_area(
                "Matrix A",
                placeholder="[[1, 2], [3, 4]]",
                key="matrix_a",
            )

            if operation in ["Matrix Addition", "Matrix Multiplication"]:
                matrix_b = st.text_area(
                    "Matrix B",
                    placeholder="[[5, 6], [7, 8]]",
                    key="matrix_b",
                )

            if st.button("Calculate", key="matrix_operation_button"):

                try:
                    a = ast.literal_eval(matrix_a)

                    if operation == "Matrix Addition":
                        result = matrix_add(a, ast.literal_eval(matrix_b))

                    elif operation == "Matrix Multiplication":
                        result = matrix_multiply(a, ast.literal_eval(matrix_b))

                    elif operation == "Matrix Transpose":
                        result = matrix_transpose(a)

                    elif operation == "Matrix Determinant":
                        result = matrix_determinant(a)

                    elif operation == "Matrix Inverse":
                        result = matrix_inverse(a)

                    elif operation == "Matrix Rank":
                        result = matrix_rank(a)

                    elif operation == "Matrix Eigenvalues":
                        result = matrix_eigenvalues(a)

                    else:
                        result = matrix_eigenvectors(a)

                    if result["success"]:
                        st.success("Result")
                        value = result["result"]

                        if operation == "Matrix Eigenvectors":
                            st.write("Eigenvalues:", value["eigenvalues"])
                            st.write("Eigenvectors:", value["eigenvectors"])
                        else:
                            st.write(value)

                    else:
                        st.error(result["error"])

                except Exception as error:
                    st.error(f"Invalid matrix input: {error}")

    # --------------------------------------------------------
    # TAB 3 — PROBABILITY & STATISTICS
    # --------------------------------------------------------

    with tab_statistics:

        from math_solver import (
            calculate_mean,
            calculate_median,
            calculate_mode,
            calculate_variance,
            calculate_standard_deviation,
            calculate_percentile,
            calculate_covariance,
            calculate_probability,
            calculate_conditional_probability,
            calculate_bayes_probability,
        )

        st.markdown("### 📊 Probability & Statistics")
        st.caption(
            "Descriptive statistics, probability, distributions, "
            "sampling, statistical tests and visualization."
        )

        stats_section = st.selectbox(
            "Choose a statistics section",
            [
                "Descriptive Statistics",
                "Probability",
                "Distributions",
                "Sampling & CLT",
                "Law of Large Numbers",
                "Statistical Tests",
                "Visualization",
            ],
            key="stats_section",
        )

        if stats_section == "Descriptive Statistics":

            operation = st.selectbox(
                "Choose operation",
                [
                    "Mean",
                    "Median",
                    "Mode",
                    "Variance",
                    "Standard Deviation",
                    "Percentile",
                    "Covariance",
                ],
                key="descriptive_operation",
            )

            data_text = st.text_input(
                "Enter data",
                placeholder="[10, 20, 20, 30, 40]",
                key="descriptive_data",
            )

            if operation == "Percentile":
                percentile = st.number_input(
                    "Percentile",
                    min_value=0.0,
                    max_value=100.0,
                    value=50.0,
                    key="percentile_value",
                )

            if operation == "Covariance":
                data_b_text = st.text_input(
                    "Enter second dataset",
                    placeholder="[12, 18, 25, 29, 35]",
                    key="covariance_data",
                )

            if st.button("Calculate", key="descriptive_button"):

                try:
                    data = ast.literal_eval(data_text)

                    functions = {
                        "Mean": calculate_mean,
                        "Median": calculate_median,
                        "Mode": calculate_mode,
                        "Variance": calculate_variance,
                        "Standard Deviation": calculate_standard_deviation,
                    }

                    if operation == "Percentile":
                        result = calculate_percentile(data, percentile)

                    elif operation == "Covariance":
                        data_b = ast.literal_eval(data_b_text)
                        result = calculate_covariance(data, data_b)

                    else:
                        result = functions[operation](data)

                    if result["success"]:
                        st.success(f"Result: {result['result']}")
                    else:
                        st.error(result["error"])

                except Exception as error:
                    st.error(f"Invalid input: {error}")

        elif stats_section == "Probability":

            operation = st.selectbox(
                "Choose operation",
                [
                    "Basic Probability",
                    "Conditional Probability",
                    "Bayes Theorem",
                ],
                key="probability_operation",
            )

            if operation == "Basic Probability":

                favorable = st.number_input(
                    "Favorable outcomes",
                    min_value=0.0,
                    key="favorable",
                )

                total = st.number_input(
                    "Total outcomes",
                    min_value=1.0,
                    key="total",
                )

                if st.button("Calculate", key="basic_probability_button"):
                    result = calculate_probability(favorable, total)

                    if result["success"]:
                        st.success(f"Probability: {result['result']}")
                    else:
                        st.error(result["error"])

            elif operation == "Conditional Probability":

                p_a_and_b = st.number_input(
                    "P(A ∩ B)",
                    min_value=0.0,
                    max_value=1.0,
                    key="p_a_and_b",
                )

                p_b = st.number_input(
                    "P(B)",
                    min_value=0.000001,
                    max_value=1.0,
                    value=0.5,
                    key="p_b",
                )

                if st.button("Calculate", key="conditional_probability_button"):
                    result = calculate_conditional_probability(p_a_and_b, p_b)

                    if result["success"]:
                        st.success(f"P(A|B): {result['result']}")
                    else:
                        st.error(result["error"])

            else:

                p_b_given_a = st.number_input(
                    "P(B|A)",
                    min_value=0.0,
                    max_value=1.0,
                    key="p_b_given_a",
                )

                p_a = st.number_input(
                    "P(A)",
                    min_value=0.0,
                    max_value=1.0,
                    key="p_a",
                )

                p_b = st.number_input(
                    "P(B)",
                    min_value=0.000001,
                    max_value=1.0,
                    value=0.5,
                    key="bayes_p_b",
                )

                if st.button("Calculate", key="bayes_button"):
                    result = calculate_bayes_probability(
                        p_b_given_a,
                        p_a,
                        p_b,
                    )

                    if result["success"]:
                        st.success(f"P(A|B): {result['result']}")
                    else:
                        st.error(result["error"])

        elif stats_section == "Distributions":

            st.selectbox(
                "Choose distribution",
                [
                    "Uniform",
                    "Bernoulli",
                    "Binomial",
                    "Geometric",
                    "Poisson",
                    "Exponential",
                    "Normal",
                    "t-Distribution",
                    "Chi-Square",
                ],
                key="distribution_type",
            )

            st.info(
                "Distribution calculations are not connected in this UI yet."
            )

        elif stats_section == "Sampling & CLT":

            st.selectbox(
                "Choose operation",
                [
                    "Random Sampling",
                    "Sample Mean",
                    "Sample Variance",
                    "Standard Error",
                    "Sampling Distribution",
                    "Central Limit Theorem",
                ],
                key="sampling_operation",
            )

            st.info(
                "Sampling and CLT calculations are not connected in this UI yet."
            )

        elif stats_section == "Law of Large Numbers":

            st.markdown("### 📈 Law of Large Numbers")

            sample_size = st.number_input(
                "Number of observations",
                min_value=10,
                value=1000,
                step=10,
                key="lln_sample_size",
            )

            st.info(
                f"LLN simulation is not connected yet. "
                f"Selected observations: {sample_size}."
            )

        elif stats_section == "Statistical Tests":

            st.selectbox(
                "Choose statistical test",
                [
                    "One-Sample t-Test",
                    "Independent t-Test",
                    "Paired t-Test",
                    "Chi-Square Goodness of Fit",
                    "Chi-Square Independence",
                ],
                key="statistical_test",
            )

            st.info("Statistical tests are not connected in this UI yet.")

        else:

            st.selectbox(
                "Choose visualization",
                [
                    "Histogram",
                    "Boxplot",
                    "Scatter Plot",
                    "Distribution Plot",
                ],
                key="visualization_type",
            )

            st.text_input(
                "Enter data",
                placeholder="[10, 12, 15, 18, 20, 21, 25]",
                key="visualization_data",
            )

            st.info("Statistical plotting is not connected in this UI yet.")


# ============================================================
# STUDY HISTORY
# ============================================================
elif mode == "📚 Study History":

    st.markdown("## 📚 Study History")
    st.info("Review your 10 most recent questions and answers.")

    display_quote()

    try:
        history = get_all_history(limit=10)

        if not history:
            st.info("No study history yet. Ask Sensei a question!")

        else:
            st.markdown(f"### 📚 Recent History ({len(history)})")

            for item in history:
                timestamp = item.get("timestamp", "")
                question = item.get("question", "Untitled question")
                answer = item.get("answer", "")

                with st.expander(f"❓ {question}"):
                    st.caption(str(timestamp))
                    st.markdown("**Answer:**")
                    st.write(answer)

    except Exception as error:
        st.error("Could not load study history.")
        st.exception(error)

# ============================================================
# POMODORO FOCUS
# ============================================================

elif mode == "🍅 Pomodoro Focus":

    st.markdown("## 🍅 Pomodoro Focus")
    st.caption("One task. One timer. One step closer to mastery.")

    st.markdown(
        "Focus deeply, take restorative breaks, and let SENSEI "
        "keep track of your completed sessions."
    )

    # --------------------------------------------------------
    # TIMER STATE
    # --------------------------------------------------------

    timer_defaults = {
        "pomodoro_type": "focus",
        "pomodoro_running": False,
        "pomodoro_end_time": None,
        "pomodoro_remaining": 25 * 60,
        "pomodoro_session_minutes": 25,
        "pomodoro_focus_streak": 0,
        "pomodoro_saved_end": None,
    }

    for key, value in timer_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    session_options = {
        "focus": ("🎯 Focus", 25),
        "short_break": ("☕ Short Break", 5),
        "long_break": ("🌿 Long Break", 15),
    }

    current_type = st.session_state.pomodoro_type

    # Refresh the page once per second while the timer runs.
    if st.session_state.pomodoro_running:
        st_autorefresh(
            interval=1000,
            key="pomodoro_second_refresh",
        )

    # --------------------------------------------------------
    # TIMER COMPLETION
    # --------------------------------------------------------

    if st.session_state.pomodoro_running:
        remaining = max(
            0,
            int(
                st.session_state.pomodoro_end_time - time.time()
                + 0.999
            ),
        )

        st.session_state.pomodoro_remaining = remaining

        if remaining == 0:
            completed_type = st.session_state.pomodoro_type
            completed_minutes = st.session_state.pomodoro_session_minutes

            # Prevent the same timer completion being saved twice.
            completed_key = st.session_state.pomodoro_end_time

            if st.session_state.pomodoro_saved_end != completed_key:
                save_pomodoro_session(
                    planned_minutes=completed_minutes,
                    completed_minutes=completed_minutes,
                    session_type=completed_type,
                )

                st.session_state.pomodoro_saved_end = completed_key

                if completed_type == "focus":
                    st.session_state.pomodoro_focus_streak += 1
                    st.session_state.pomodoro_type = (
                        "long_break"
                        if st.session_state.pomodoro_focus_streak % 4 == 0
                        else "short_break"
                    )

            st.session_state.pomodoro_running = False
            st.session_state.pomodoro_end_time = None

            next_type = st.session_state.pomodoro_type
            next_minutes = session_options[next_type][1]
            st.session_state.pomodoro_remaining = next_minutes * 60
            st.session_state.pomodoro_session_minutes = next_minutes

            st.success("🎉 Session completed! Beautiful work.")
            st.rerun()

    # --------------------------------------------------------
    # SESSION SELECTOR
    # --------------------------------------------------------

    if not st.session_state.pomodoro_running:
        chosen_type = st.radio(
            "Choose your session",
            options=list(session_options.keys()),
            format_func=lambda item: session_options[item][0],
            horizontal=True,
            index=list(session_options.keys()).index(
                st.session_state.pomodoro_type
            ),
            key="pomodoro_session_selector",
        )

        if chosen_type != st.session_state.pomodoro_type:
            st.session_state.pomodoro_type = chosen_type
            st.session_state.pomodoro_session_minutes = (
                session_options[chosen_type][1]
            )
            st.session_state.pomodoro_remaining = (
                session_options[chosen_type][1] * 60
            )

    current_type = st.session_state.pomodoro_type

    # --------------------------------------------------------
    # TIMER DISPLAY
    # --------------------------------------------------------

    remaining = st.session_state.pomodoro_remaining
    minutes, seconds = divmod(remaining, 60)

    st.divider()
    st.markdown(
        f"<div style='text-align:center; padding:24px 8px;'>"
        f"<div style='font-size:18px; opacity:0.8;'>"
        f"{session_options[current_type][0]}</div>"
        f"<div style='font-size:clamp(54px, 10vw, 88px); "
        f"font-weight:750; letter-spacing:3px;'>"
        f"{minutes:02d}:{seconds:02d}</div>"
        f"<div style='opacity:0.7;'>"
        f"{'Stay present. You have got this.' if current_type == 'focus' else 'Rest is part of the process.'}"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # TIMER CONTROLS
    # --------------------------------------------------------

    col_start, col_pause, col_reset = st.columns(3)

    with col_start:
        if st.button(
            "▶ Start / Resume",
            use_container_width=True,
            disabled=st.session_state.pomodoro_running,
            key="pomodoro_start",
        ):
            st.session_state.pomodoro_end_time = (
                time.time() + st.session_state.pomodoro_remaining
            )
            st.session_state.pomodoro_running = True
            st.rerun()

    with col_pause:
        if st.button(
            "⏸ Pause",
            use_container_width=True,
            disabled=not st.session_state.pomodoro_running,
            key="pomodoro_pause",
        ):
            st.session_state.pomodoro_remaining = max(
                0,
                int(
                    st.session_state.pomodoro_end_time
                    - time.time()
                    + 0.999
                ),
            )
            st.session_state.pomodoro_running = False
            st.session_state.pomodoro_end_time = None
            st.rerun()

    with col_reset:
        if st.button(
            "↻ Reset",
            use_container_width=True,
            key="pomodoro_reset",
        ):
            st.session_state.pomodoro_running = False
            st.session_state.pomodoro_end_time = None
            st.session_state.pomodoro_remaining = (
                st.session_state.pomodoro_session_minutes * 60
            )
            st.rerun()

    st.caption(
        "Completed sessions are saved to your local database. "
        "Paused or reset sessions are not counted as completed."
    )

    # --------------------------------------------------------
    # PERSISTENT STATISTICS
    # --------------------------------------------------------

    stats = get_pomodoro_stats()

    st.divider()
    st.markdown("### 🏆 Your focus journey")

    stat1, stat2, stat3 = st.columns(3)

    stat1.metric(
        "Focus sessions",
        stats["completed_focus_sessions"],
    )
    stat2.metric(
        "Focus minutes",
        stats["focus_minutes"],
    )
    stat3.metric(
        "Completed breaks",
        stats["completed_breaks"],
    )

    st.markdown("### 📜 Recent sessions")

    recent_sessions = get_recent_pomodoro_sessions(limit=10)

    if recent_sessions:
        for session in recent_sessions:
            st.markdown(
                f"- **{session['session_type'].replace('_', ' ').title()}** "
                f"— {session['completed_minutes']} min "
                f"· {session['completed_at']}"
            )
    else:
        st.info("Your first completed session will appear here.")


# ============================================================
# TO-DO LIST
# ============================================================

elif mode == "✨ To-do List":

    st.markdown("## ✨ Your Dojo To-do List")
    st.caption("Small wins. Consistent effort. Real progress.")

    # --------------------------------------------------------
    # ADD TASK
    # --------------------------------------------------------

    with st.form("sensei_add_todo_form", clear_on_submit=True):
        task_title = st.text_input(
            "What do you want to accomplish?",
            placeholder="e.g. Revise Dijkstra's algorithm",
        )

        col_category, col_priority = st.columns(2)

        with col_category:
            task_category = st.selectbox(
                "Category",
                [
                    "College",
                    "AI / ML",
                    "DSA",
                    "Projects",
                    "Personal",
                    "General",
                ],
            )

        with col_priority:
            task_priority = st.selectbox(
                "Priority",
                ["High", "Medium", "Low"],
                index=1,
            )

        add_task = st.form_submit_button(
            "＋ Add task",
            type="primary",
            use_container_width=True,
        )

    if add_task:
        if not task_title.strip():
            st.warning("Give your task a title first.")
        else:
            add_todo_task(
                title=task_title,
                category=task_category,
                priority=task_priority,
            )
            st.success("Task added to your list!")
            st.rerun()

    # --------------------------------------------------------
    # TASK FILTER
    # --------------------------------------------------------

    st.divider()

    task_view = st.radio(
        "Show tasks",
        ["All", "Pending", "Completed"],
        horizontal=True,
        key="sensei_todo_filter",
    )

    tasks = get_todo_tasks(include_completed=True)

    if task_view == "Pending":
        tasks = [
            task for task in tasks
            if not task["is_completed"]
        ]
    elif task_view == "Completed":
        tasks = [
            task for task in tasks
            if task["is_completed"]
        ]

    all_tasks = get_todo_tasks(include_completed=True)
    total_tasks = len(all_tasks)
    completed_tasks = sum(
        bool(task["is_completed"]) for task in all_tasks
    )
    pending_tasks = total_tasks - completed_tasks

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Total", total_tasks)
    metric2.metric("Pending", pending_tasks)
    metric3.metric("Completed", completed_tasks)

    if total_tasks:
        st.progress(completed_tasks / total_tasks)
        st.caption(
            f"{completed_tasks} of {total_tasks} tasks completed."
        )

    # --------------------------------------------------------
    # DISPLAY TASKS
    # --------------------------------------------------------

    st.markdown("### 📋 Your tasks")

    if not tasks:
        st.info(
            "Nothing here yet. Add a task above and start collecting wins! 🌱"
        )

    for task in tasks:
        task_id = task["id"]
        is_completed = bool(task["is_completed"])

        with st.container(border=True):
            col_check, col_details, col_delete = st.columns(
                [0.7, 4, 0.8]
            )

            with col_check:
                if st.checkbox(
                    "Done",
                    value=is_completed,
                    key=f"todo_done_{task_id}",
                    label_visibility="collapsed",
                ) != is_completed:
                    set_todo_completed(
                        task_id,
                        not is_completed,
                    )
                    st.rerun()

            with col_details:
                title = task["title"]

                if is_completed:
                    st.markdown(f"~~{title}~~")
                else:
                    st.markdown(f"**{title}**")

                st.caption(
                    f"{task['category']} · "
                    f"{task['priority']} priority"
                )

                if task.get("completed_at"):
                    st.caption(
                        f"Completed: {task['completed_at']}"
                    )

            with col_delete:
                if st.button(
                    "🗑️",
                    key=f"todo_delete_{task_id}",
                    help="Delete this task",
                ):
                    delete_todo_task(task_id)
                    st.rerun()
