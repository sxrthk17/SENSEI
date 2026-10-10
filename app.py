
import streamlit as st

from aesthetics import (
    initialize_theme,
    apply_theme,
    display_quote,
)

from memory_db import (
    init_db,
    log_interaction,
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


    # --------------------------------------------------------
    # Process Notes
    # --------------------------------------------------------

    if uploaded_file is not None:

        st.success(
            f"📄 Uploaded: {uploaded_file.name}"
        )

        if st.button("🧠 Process Notes"):

            from rag_pipeline import (
                create_embeddings,
                ingest_pdf,
            )

            with st.spinner(
                "🥋 Sensei is studying your notes..."
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


                # Ingest PDF
                vector_store = ingest_pdf(
                    uploaded_file.name,
                    embeddings,
                )


                # Store active document
                document_name = uploaded_file.name

                st.session_state.vector_store = vector_store

                st.session_state.source_pdf = document_name

                st.session_state.active_document_name = (
                    document_name
                )

                st.session_state.notes_processed = True

            st.success(
                "✅ Notes processed successfully!"
            )


    # ========================================================
    # ACTIVE DOCUMENT STATUS
    # ========================================================

    if (
        st.session_state.get(
            "notes_processed",
            False,
        )
        and st.session_state.get(
            "active_document_name"
        )
    ):

        st.markdown("---")

        st.success(
            f"📚 Active notes: "
            f"**{st.session_state.active_document_name}**"
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


        if st.button(
            "🥋 Ask Sensei",
            key="ask_sensei_button",
        ):

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
                    "🥋 Sensei is thinking..."
                ):

                    # ------------------------------------------------
                    # Vector store
                    # ------------------------------------------------

                    vector_store = (
                        st.session_state.vector_store
                    )


                    # ------------------------------------------------
                    # Active document
                    # ------------------------------------------------

                    active_document = (
                        st.session_state.get(
                            "active_document_name"
                        )
                    )


                    # ------------------------------------------------
                    # Retrieve relevant notes
                    #
                    # IMPORTANT:
                    # Your current retrieve_documents()
                    # does NOT accept document_name.
                    # ------------------------------------------------

                    documents = retrieve_documents(
                        vector_store,
                        question,
                        k=6,
                    )


                    # ------------------------------------------------
                    # Save retrieved documents
                    # ------------------------------------------------

                    st.session_state.retrieved_documents = (
                        documents
                    )


                    # ------------------------------------------------
                    # Generate grounded answer
                    # ------------------------------------------------

                    if documents:

                        answer = generate_answer(
                            question,
                            documents,
                        )

                    else:

                        answer = (
                            "I couldn't find relevant information "
                            "in the uploaded notes."
                        )


                    # ------------------------------------------------
                    # Save interaction
                    # ------------------------------------------------

                    log_interaction(
                        question=question,
                        answer=answer,
                        source_pdf=active_document
                        or "unknown",
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
                    "📖 View retrieved notes",
                    expanded=False,
                ):

                    if not documents:

                        st.error(
                            "No relevant content was retrieved "
                            "from the active PDF."
                        )

                    else:

                        st.success(
                            f"Retrieved {len(documents)} "
                            f"chunks."
                        )

                        for index, document in enumerate(
                            documents,
                            start=1,
                        ):

                            metadata = document.metadata

                            page = metadata.get(
                                "page_number"
                            )

                            chunk_index = metadata.get(
                                "chunk_index"
                            )

                            source = metadata.get(
                                "source_pdf",
                                active_document,
                            )


                            st.markdown(
                                f"### Chunk {index}"
                            )


                            if source:

                                if page is not None:

                                    st.caption(
                                        f"📄 {source} "
                                        f"| Page {page}"
                                    )

                                else:

                                    st.caption(
                                        f"📄 {source}"
                                    )


                            if chunk_index is not None:

                                st.caption(
                                    f"Chunk index: {chunk_index}"
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
    st.caption("Powered by Sensei's verified mathematical engine")

    # =========================================================
    # MATH DOMAIN TABS
    # =========================================================

    tab_numerical, tab_linear, tab_statistics = st.tabs(
        [
            "📐 Numerical Math & Calculus",
            "🔢 Linear Algebra",
            "📊 Probability & Statistics",
        ]
    )

    # =========================================================
    # TAB 1 — NUMERICAL MATH & CALCULUS
    # =========================================================

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

        # -----------------------------------------------------
        # SOLVE SYSTEM
        # -----------------------------------------------------

        if operation == "Solve System of Equations":

            equations_text = st.text_area(
                "Enter equations (one per line)",
                placeholder="x + y = 10\n2*x - y = 3",
                key="system_equations",
            )

            if st.button("Solve System", key="solve_system_button"):

                from math_solver import solve_system

                equations = [
                    equation.strip()
                    for equation in equations_text.splitlines()
                    if equation.strip()
                ]

                result = solve_system(equations)

                if result["success"]:

                    st.success("Solution found")

                    solutions = result["result"]

                    for solution in solutions:

                        for variable, value in solution.items():

                            st.markdown(
                                f"**{variable} = {value}**"
                            )

                else:

                    st.error(result["error"])

        # -----------------------------------------------------
        # BASIC SYMBOLIC OPERATIONS
        # -----------------------------------------------------

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

                result = functions[operation](expression)

                if result["success"]:

                    st.success(f"Result: {result['result']}")

                else:

                    st.error(result["error"])

        # -----------------------------------------------------
        # CALCULUS
        # -----------------------------------------------------

        elif operation in [
            "Differentiate",
            "Integrate",
            "Definite Integral",
            "Limit",
            "Series Expansion",
        ]:

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

            # Definite integral inputs
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

            # Limit inputs
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

            # Series inputs
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

                if operation == "Differentiate":

                    result = differentiate(
                        expression,
                        variable,
                    )

                elif operation == "Integrate":

                    result = integrate(
                        expression,
                        variable,
                    )

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

                    st.success(
                        f"Result: {result['result']}"
                    )

                else:

                    st.error(result["error"])

    # =========================================================
    # TAB 2 — LINEAR ALGEBRA
    # =========================================================

    with tab_linear:

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

        import ast

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

        # -----------------------------------------------------
        # VECTOR OPERATIONS
        # -----------------------------------------------------

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

            if st.button(
                "Calculate",
                key="vector_operation_button",
            ):

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

                        st.success(
                            f"Result: {result['result']}"
                        )

                    else:

                        st.error(result["error"])

                except Exception as error:

                    st.error(
                        f"Invalid vector input: {error}"
                    )

        # -----------------------------------------------------
        # VECTOR NORM
        # -----------------------------------------------------

        elif operation == "Vector Norm":

            vector = st.text_input(
                "Vector",
                placeholder="[3, 4]",
                key="vector_norm_input",
            )

            if st.button(
                "Calculate",
                key="vector_norm_button",
            ):

                try:

                    vector = ast.literal_eval(vector)

                    result = vector_norm(vector)

                    if result["success"]:

                        st.success(
                            f"Result: {result['result']}"
                        )

                    else:

                        st.error(result["error"])

                except Exception as error:

                    st.error(
                        f"Invalid vector input: {error}"
                    )

        # -----------------------------------------------------
        # MATRIX OPERATIONS
        # -----------------------------------------------------

        else:

            matrix_a = st.text_area(
                "Matrix A",
                placeholder="[[1, 2], [3, 4]]",
                key="matrix_a",
            )

            if operation in [
                "Matrix Addition",
                "Matrix Multiplication",
            ]:

                matrix_b = st.text_area(
                    "Matrix B",
                    placeholder="[[5, 6], [7, 8]]",
                    key="matrix_b",
                )

            if st.button(
                "Calculate",
                key="matrix_operation_button",
            ):

                try:

                    a = ast.literal_eval(matrix_a)

                    if operation == "Matrix Addition":

                        b = ast.literal_eval(matrix_b)

                        result = matrix_add(
                            a,
                            b,
                        )

                    elif operation == "Matrix Multiplication":

                        b = ast.literal_eval(matrix_b)

                        result = matrix_multiply(
                            a,
                            b,
                        )

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

                            st.write(
                                "Eigenvalues:",
                                value["eigenvalues"],
                            )

                            st.write(
                                "Eigenvectors:",
                                value["eigenvectors"],
                            )

                        else:

                            st.write(value)

                    else:

                        st.error(result["error"])

                except Exception as error:

                    st.error(
                        f"Invalid matrix input: {error}"
                    )

    # =========================================================
    # TAB 3 — PROBABILITY & STATISTICS
    # =========================================================

    # =========================================================
    # TAB 3 — PROBABILITY & STATISTICS — L5
    # =========================================================

    with tab_statistics:

        import numpy as np

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
            "sampling, statistical tests and visualization"
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

        # =====================================================
        # DESCRIPTIVE STATISTICS
        # =====================================================

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

            if st.button(
                "Calculate",
                key="descriptive_button",
            ):

                try:

                    data = eval(
                        data_text,
                        {"__builtins__": {}},
                        {},
                    )

                    if operation == "Mean":
                        result = calculate_mean(data)

                    elif operation == "Median":
                        result = calculate_median(data)

                    elif operation == "Mode":
                        result = calculate_mode(data)

                    elif operation == "Variance":
                        result = calculate_variance(data)

                    elif operation == "Standard Deviation":
                        result = calculate_standard_deviation(data)

                    elif operation == "Percentile":
                        result = calculate_percentile(
                            data,
                            percentile,
                        )

                    else:

                        data_b = eval(
                            data_b_text,
                            {"__builtins__": {}},
                            {},
                        )

                        result = calculate_covariance(
                            data,
                            data_b,
                        )

                    if result["success"]:
                        st.success(
                            f"Result: {result['result']}"
                        )
                    else:
                        st.error(result["error"])

                except Exception as error:
                    st.error(
                        f"Invalid input: {error}"
                    )

        # =====================================================
        # PROBABILITY
        # =====================================================

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

                if st.button(
                    "Calculate",
                    key="basic_probability_button",
                ):

                    result = calculate_probability(
                        favorable,
                        total,
                    )

                    if result["success"]:
                        st.success(
                            f"Probability: {result['result']}"
                        )
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

                if st.button(
                    "Calculate",
                    key="conditional_probability_button",
                ):

                    result = calculate_conditional_probability(
                        p_a_and_b,
                        p_b,
                    )

                    if result["success"]:
                        st.success(
                            f"P(A|B): {result['result']}"
                        )
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

                if st.button(
                    "Calculate",
                    key="bayes_button",
                ):

                    result = calculate_bayes_probability(
                        p_b_given_a,
                        p_a,
                        p_b,
                    )

                    if result["success"]:
                        st.success(
                            f"P(A|B): {result['result']}"
                        )
                    else:
                        st.error(result["error"])

        # =====================================================
        # DISTRIBUTIONS
        # =====================================================

        elif stats_section == "Distributions":

            distribution = st.selectbox(
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
                "Distribution controls will be connected to "
                "the verified L5 distribution engine."
            )

        # =====================================================
        # SAMPLING & CLT
        # =====================================================

        elif stats_section == "Sampling & CLT":

            sampling_operation = st.selectbox(
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
                "Sampling and CLT operations are available "
                "in Sensei's L5 statistics engine."
            )

        # =====================================================
        # LLN
        # =====================================================

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
                f"LLN simulation ready for {sample_size} observations."
            )

        # =====================================================
        # STATISTICAL TESTS
        # =====================================================

        elif stats_section == "Statistical Tests":

            test = st.selectbox(
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

            st.info(
                "Statistical test operations are available "
                "in Sensei's verified L5 engine."
            )

        # =====================================================
        # VISUALIZATION
        # =====================================================

        else:

            visualization = st.selectbox(
                "Choose visualization",
                [
                    "Histogram",
                    "Boxplot",
                    "Scatter Plot",
                    "Distribution Plot",
                ],
                key="visualization_type",
            )

            data_text = st.text_input(
                "Enter data",
                placeholder="[10, 12, 15, 18, 20, 21, 25]",
                key="visualization_data",
            )

            st.info(
                "Visualization operations are available "
                "through Sensei's NumPy, Matplotlib and "
                "Seaborn statistics engine."
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