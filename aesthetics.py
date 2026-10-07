from pathlib import Path
import json
import random

import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
QUOTES_FILE = ASSETS_DIR / "quotes.json"


# ============================================================
# Visual Themes
# ============================================================

THEMES = [
    {
        "name": "Midnight Dojo",
        "background": "#0F172A",
        "text": "#F8FAFC",
        "accent": "#38BDF8",
    },
    {
        "name": "Dark Academy",
        "background": "#18181B",
        "text": "#FAFAFA",
        "accent": "#A78BFA",
    },
    {
        "name": "Forest Training",
        "background": "#102019",
        "text": "#ECFDF5",
        "accent": "#34D399",
    },
    {
        "name": "Ancient Scroll",
        "background": "#292218",
        "text": "#FEF3C7",
        "accent": "#F59E0B",
    },
]


# ============================================================
# Mascot Evolution
# ============================================================

MASCOTS = {
    1: {
        "name": "Apprentice",
        "file": "mascot_01.svg",
        "description": "The journey begins.",
    },
    2: {
        "name": "Student",
        "file": "mascot_02.svg",
        "description": "Learning becomes discipline.",
    },
    3: {
        "name": "Practitioner",
        "file": "mascot_03.svg",
        "description": "Knowledge becomes practice.",
    },
    4: {
        "name": "Scholar",
        "file": "mascot_04.svg",
        "description": "Understanding begins to deepen.",
    },
    5: {
        "name": "Master",
        "file": "mascot_05.svg",
        "description": "Skill becomes instinct.",
    },
    6: {
        "name": "Sensei",
        "file": "mascot_06.svg",
        "description": "Mastery becomes something you can teach.",
    },
}


# ============================================================
# Quotes
# ============================================================

@st.cache_data(show_spinner=False)
def load_quotes():
    """Load the original Sensei quote bank."""

    if not QUOTES_FILE.exists():
        return ["Keep learning. Keep building."]

    try:
        with open(QUOTES_FILE, "r", encoding="utf-8") as file:
            quotes = json.load(file)
    except (OSError, json.JSONDecodeError):
        return ["Keep learning. Keep building."]

    if not isinstance(quotes, list) or not quotes:
        return ["Keep learning. Keep building."]

    return quotes


def get_random_quote():
    """Return one random quote from the Sensei quote bank."""

    quotes = load_quotes()
    return random.choice(quotes)


# ============================================================
# Themes
# ============================================================

def get_random_theme():
    """Return one random visual theme."""

    return random.choice(THEMES)


def initialize_theme():
    """
    Select a theme once for the current Streamlit session.

    Streamlit reruns the script frequently, so session_state
    prevents the background from changing on every interaction.
    """

    if "sensei_theme" not in st.session_state:
        st.session_state.sensei_theme = get_random_theme()

    return st.session_state.sensei_theme


# ============================================================
# Mascots
# ============================================================

def get_mascot(stage=1):
    """Return mascot information for the requested evolution stage."""

    stage = max(1, min(stage, len(MASCOTS)))

    mascot = MASCOTS[stage].copy()
    mascot["path"] = ASSETS_DIR / mascot["file"]

    return mascot


def get_mascot_svg(stage=1):
    """Read and return the mascot SVG."""

    mascot = get_mascot(stage)
    mascot_path = mascot["path"]

    if not mascot_path.exists():
        return None

    try:
        return mascot_path.read_text(encoding="utf-8")
    except OSError:
        return None


# ============================================================
# Global Sensei Styling
# ============================================================

def apply_theme(theme):
    """Apply the selected Sensei theme to the Streamlit app."""

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-color: {theme["background"]};
            color: {theme["text"]};
        }}

        .sensei-accent {{
            color: {theme["accent"]};
        }}

        .sensei-card {{
            padding: 1rem;
            border-radius: 14px;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(255, 255, 255, 0.05);
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Quote UI
# ============================================================

def display_quote(quote=None):
    """Display a Sensei quote card."""

    if quote is None:
        quote = get_random_quote()

    st.markdown(
        f"""
        <div class="sensei-card" style="text-align: center;">
            <em>“{quote}”</em>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Mascot UI
# ============================================================

def display_mascot(stage=1):
    """Display a mascot and its evolution information."""

    mascot = get_mascot(stage)
    svg = get_mascot_svg(stage)

    if svg is not None:
        components.html(
            svg,
            height=450,
            scrolling=False,
        )
    else:
        st.warning(f"Could not load {mascot['file']}.")

    st.markdown(
        f"""
        <div style="text-align: center;">
            <h3 class="sensei-accent">{mascot["name"]}</h3>
            <p>{mascot["description"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )