
from pathlib import Path
import json
import random
import html

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
# Cyan remains the consistent Sensei brand accent.
# ============================================================

THEMES = [
    {
        "name": "Midnight Dojo",
        "background": "#0B1220",
        "surface": "#111C2E",
        "text": "#F1F5F9",
        "muted": "#94A3B8",
        "accent": "#38BDF8",
    },
    {
        "name": "Dark Academy",
        "background": "#111318",
        "surface": "#1B2029",
        "text": "#F3F4F6",
        "muted": "#9CA3AF",
        "accent": "#38BDF8",
    },
    {
        "name": "Forest Training",
        "background": "#0D1B17",
        "surface": "#142720",
        "text": "#ECFDF5",
        "muted": "#A7C4B6",
        "accent": "#38BDF8",
    },
    {
        "name": "Ancient Scroll",
        "background": "#211D19",
        "surface": "#302820",
        "text": "#FEF3C7",
        "muted": "#C7B99F",
        "accent": "#38BDF8",
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
        with QUOTES_FILE.open("r", encoding="utf-8") as file:
            quotes = json.load(file)
    except (OSError, json.JSONDecodeError):
        return ["Keep learning. Keep building."]

    if not isinstance(quotes, list) or not quotes:
        return ["Keep learning. Keep building."]

    return [quote for quote in quotes if isinstance(quote, str)] or [
        "Keep learning. Keep building."
    ]


def get_random_quote():
    return random.choice(load_quotes())


# ============================================================
# Theme Management
# ============================================================

def get_random_theme():
    return random.choice(THEMES)


def get_theme_by_name(name):
    """Return a theme by its displayed name."""

    return next(
        (theme for theme in THEMES if theme["name"] == name),
        THEMES[0],
    )


def initialize_theme():
    """
    Preserve the chosen theme across Streamlit reruns.
    A random theme is selected only for a new session.
    """

    if "sensei_theme" not in st.session_state:
        st.session_state.sensei_theme = get_random_theme()

    current = st.session_state.sensei_theme

    # Support both a theme dictionary and a saved theme name.
    if isinstance(current, str):
        current = get_theme_by_name(current)
        st.session_state.sensei_theme = current

    return current


def render_theme_selector():
    """Render the manual theme selector in the sidebar."""

    current = initialize_theme()
    theme_names = [theme["name"] for theme in THEMES]

    current_name = (
        current["name"]
        if current["name"] in theme_names
        else theme_names[0]
    )

    selected_name = st.selectbox(
        "Workspace theme",
        options=theme_names,
        index=theme_names.index(current_name),
        key="sensei_theme_selector",
    )

    selected_theme = get_theme_by_name(selected_name)

    if selected_theme["name"] != current["name"]:
        st.session_state.sensei_theme = selected_theme
        st.rerun()

    return selected_theme


# ============================================================
# Mascots
# ============================================================

def get_mascot(stage=1):
    stage = max(1, min(int(stage), len(MASCOTS)))

    mascot = MASCOTS[stage].copy()
    mascot["path"] = ASSETS_DIR / mascot["file"]

    return mascot


def get_mascot_svg(stage=1):
    mascot = get_mascot(stage)
    path = mascot["path"]

    if not path.exists():
        return None

    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


# ============================================================
# Global Styling
# ============================================================

def apply_theme(theme):
    """Apply the selected theme and shared workspace styling."""

    if isinstance(theme, str):
        theme = get_theme_by_name(theme)

    background = theme["background"]
    surface = theme.get("surface", "#172033")
    text = theme["text"]
    muted = theme.get("muted", "#94A3B8")
    accent = theme["accent"]

    st.markdown(
        f"""
        <style>
        :root {{
            --sensei-bg: {background};
            --sensei-surface: {surface};
            --sensei-text: {text};
            --sensei-muted: {muted};
            --sensei-accent: {accent};
            --sensei-border: rgba(148, 163, 184, 0.18);
        }}

        .stApp {{
            background: var(--sensei-bg);
            color: var(--sensei-text);
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        [data-testid="stSidebar"] {{
            background: var(--sensei-surface);
            border-right: 1px solid var(--sensei-border);
        }}

        [data-testid="stSidebar"] > div:first-child {{
            background: var(--sensei-surface);
        }}

        [data-testid="stSidebar"] .block-container {{
            padding-top: 1rem;
            padding-bottom: 1.5rem;
        }}

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label {{
            color: var(--sensei-text);
        }}

        [data-testid="stSidebar"] hr {{
            border-color: var(--sensei-border);
            margin: 1rem 0;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] {{
            gap: 0.35rem;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label {{
            border-radius: 9px;
            padding: 0.25rem 0.4rem;
        }}

        .block-container {{
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1440px;
        }}

        h1, h2, h3, h4 {{
            color: var(--sensei-text);
            letter-spacing: -0.025em;
        }}

        p, li, label {{
            line-height: 1.65;
        }}

        .sensei-accent {{
            color: var(--sensei-accent) !important;
        }}

        .sensei-eyebrow {{
            color: var(--sensei-accent);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }}

        .sensei-hero {{
            padding: 0.75rem 0 0.5rem;
        }}

        .sensei-hero h1 {{
            font-size: clamp(2rem, 3.5vw, 3rem);
            font-weight: 800;
            line-height: 1.12;
            margin-bottom: 0.5rem;
        }}

        .sensei-subtitle {{
            color: var(--sensei-muted);
            font-size: 1rem;
            line-height: 1.6;
            max-width: 680px;
        }}

        .sensei-card {{
            padding: 1.2rem 1.35rem;
            border-radius: 16px;
            margin: 0.85rem 0;
            border: 1px solid var(--sensei-border);
            background: var(--sensei-surface);
            overflow-wrap: anywhere;
            transition: border-color 160ms ease;
        }}

        .sensei-card p:last-child {{
            margin-bottom: 0;
        }}

        .sensei-card:hover {{
            border-color: rgba(56, 189, 248, 0.4);
        }}

        .sensei-quote {{
            border-left: 3px solid var(--sensei-accent);
            padding: 0.85rem 1rem;
            border-radius: 0 12px 12px 0;
            background: rgba(56, 189, 248, 0.055);
            color: var(--sensei-text);
            margin: 1rem 0;
        }}

        .sensei-muted {{
            color: var(--sensei-muted);
        }}

        .sensei-divider {{
            border: 0;
            border-top: 1px solid var(--sensei-border);
            margin: 1.25rem 0;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border-color: var(--sensei-border);
            border-radius: 16px;
        }}

        div.stButton > button {{
            border-radius: 10px;
            border: 1px solid var(--sensei-border);
            font-weight: 600;
            transition: border-color 150ms ease,
                        transform 150ms ease;
        }}

        div.stButton > button:hover {{
            border-color: var(--sensei-accent);
            transform: translateY(-1px);
        }}

        div.stButton > button[kind="primary"] {{
            background: var(--sensei-accent);
            color: #07111D;
            border-color: var(--sensei-accent);
        }}

        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        div[data-baseweb="select"] > div {{
            background: var(--sensei-surface);
            border-color: var(--sensei-border);
            border-radius: 10px;
        }}

        [data-testid="stMetric"] {{
            background: var(--sensei-surface);
            border: 1px solid var(--sensei-border);
            border-radius: 14px;
            padding: 0.85rem 1rem;
        }}

        [data-testid="stExpander"] {{
            border-color: var(--sensei-border);
            border-radius: 12px;
        }}

        [data-testid="stFileUploader"] {{
            border-radius: 14px;
        }}

        [data-testid="stCaptionContainer"] {{
            color: var(--sensei-muted);
        }}

        ::selection {{
            background: var(--sensei-accent);
            color: #07111D;
        }}

        @media (max-width: 768px) {{
            .block-container {{
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }}

            .sensei-card {{
                padding: 1rem;
            }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            div.stButton > button,
            .sensei-card {{
                transition: none;
            }}

            div.stButton > button:hover {{
                transform: none;
            }}
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

    safe_quote = html.escape(str(quote))

    st.markdown(
        f"""
        <div class="sensei-quote">
            <span class="sensei-eyebrow">A thought for today</span>
            <div><em>“{safe_quote}”</em></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Mascot UI
# ============================================================
def display_mascot(stage=6):
    """Display an existing Sensei mascot in a responsive animated frame."""

    mascot = get_mascot(stage)
    svg = get_mascot_svg(stage)

    if not svg:
        st.warning(
            f"Could not load {mascot['file']}. "
            f"Check that it exists inside: {ASSETS_DIR}"
        )
        return

    safe_name = html.escape(mascot["name"])
    safe_description = html.escape(mascot["description"])

    components.html(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="UTF-8">
        <style>
            * {{
                box-sizing: border-box;
            }}

            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                background: transparent;
                overflow: hidden;
                font-family: Arial, sans-serif;
            }}

            .mascot-scene {{
                width: 100%;
                height: 300px;
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
                isolation: isolate;
            }}

            .mascot-glow {{
                position: absolute;
                width: 210px;
                height: 210px;
                border-radius: 50%;
                background: radial-gradient(
                    circle,
                    rgba(56, 189, 248, 0.22) 0%,
                    rgba(56, 189, 248, 0.07) 48%,
                    transparent 72%
                );
                filter: blur(8px);
                animation: breathe 4s ease-in-out infinite;
                z-index: -1;
            }}

            .mascot-art {{
                width: 100%;
                height: 270px;
                display: flex;
                align-items: center;
                justify-content: center;
                animation: float 4s ease-in-out infinite;
            }}

            .mascot-art svg {{
                display: block;
                width: 100%;
                height: 100%;
                max-width: 280px;
                max-height: 270px;
                object-fit: contain;
                overflow: visible;
            }}

            @keyframes float {{
                0%, 100% {{
                    transform: translateY(0);
                }}
                50% {{
                    transform: translateY(-8px);
                }}
            }}

            @keyframes breathe {{
                0%, 100% {{
                    opacity: 0.55;
                    transform: scale(0.94);
                }}
                50% {{
                    opacity: 1;
                    transform: scale(1.08);
                }}
            }}

            @media (prefers-reduced-motion: reduce) {{
                .mascot-art, .mascot-glow {{
                    animation: none;
                }}
            }}
        </style>
        </head>
        <body>
            <div class="mascot-scene">
                <div class="mascot-glow"></div>
                <div class="mascot-art">
                    {svg}
                </div>
            </div>
        </body>
        </html>
        """,
        height=310,
        scrolling=False,
    )

    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding:0 0 1rem;
            margin-top:-0.4rem;
        ">
            <div style="
                color:#38BDF8;
                font-size:1.35rem;
                font-weight:800;
                letter-spacing:0.04em;
            ">{safe_name}</div>
            <div style="
                opacity:0.72;
                font-size:0.9rem;
                margin-top:0.3rem;
            ">{safe_description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
