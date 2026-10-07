from pathlib import Path
import json
import random

import streamlit as st
import streamlit.components.v1 as components


from aesthetics import (
    initialize_theme,
    apply_theme,
    display_quote,
    display_mascot,
)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Sensei — Aesthetic Preview",
    page_icon="🥋",
    layout="centered",
)


# ============================================================
# Initialize Sensei Aesthetics
# ============================================================

theme = initialize_theme()

apply_theme(theme)


# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div style="text-align: center;">
        <h1 class="sensei-accent">🥋 SENSEI</h1>
        <p>Aesthetic Preview</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Theme Information
# ============================================================

st.markdown(
    f"""
    <div class="sensei-card" style="text-align: center;">
        <strong>Current Theme</strong>
        <br>
        {theme["name"]}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Mascot Preview
# ============================================================

st.markdown("### Mascot Evolution")

stage = st.slider(
    "Choose evolution stage",
    min_value=1,
    max_value=6,
    value=6,
)

display_mascot(stage)


# ============================================================
# Quote Preview
# ============================================================

st.markdown("### Sensei Quote")

display_quote()


# ============================================================
# Theme Preview
# ============================================================

st.markdown(
    """
    <div class="sensei-card">
        <strong>Theme System</strong>
        <br><br>
        The theme is selected once when the session begins.
        Streamlit reruns will not randomly change it.
    </div>
    """,
    unsafe_allow_html=True,
)