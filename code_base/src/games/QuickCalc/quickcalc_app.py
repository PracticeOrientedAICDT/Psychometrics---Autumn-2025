import streamlit as st
import time
import numpy as np
import pandas as pd

from engine import QuickCalcEngine
from mechanics import load_mechanics
from utils import clamp, load_image_base64

# ------------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="QuickCalc",
    page_icon="🎈",
    layout="wide",
)

# ------------------------------------------------------------
# BALLOON IMAGE (base64 embedded fallback)
# ------------------------------------------------------------
BALLOON_IMG = load_image_base64()

# ------------------------------------------------------------
# HELPER: RENDER A BALLOON AS HTML
# ------------------------------------------------------------
def render_balloon_html(balloon, y_pct, channel_width=200, balloon_size=50):
    """
    balloon: Balloon instance
    y_pct: float 0..1 (0 at top, 1 at bottom)
    """
    top_offset = (1 - y_pct) * 300  # px movement

    html = f"""
    <div style="
        position:absolute;
        left:0;
        right:0;
        margin:auto;
        width:{balloon_size}px;
        height:{balloon_size}px;
        top:{top_offset}px;
        text-align:center;
    ">
        <img src="data:image/png;base64,{BALLOON_IMG}"
             width="{balloon_size}"
             style="filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.4));"/>
        <div style="font-size:16px; font-weight:600;">
            {balloon.left} {balloon.operation} {balloon.right}
        </div>
    </div>
    """
    return html

# ------------------------------------------------------------
# ON-SCREEN KEYPAD
# ------------------------------------------------------------
def render_keypad():
    cols = st.columns(3)
    keys = [
        ("1","2","3"),
        ("4","5","6"),
        ("7","8","9"),
        ("0","⌫","Enter"),
    ]
    for row in keys:
        c = st.columns(3)
        for i, key in enumerate(row):
            if c[i].button(key, use_container_width=True):
                return key
    return None

# ------------------------------------------------------------
# SESSION INITIALISATION
# ------------------------------------------------------------
if "engine" not in st.session_state:
    st.session_state.engine = None

if "answer_buffer" not in st.session_state:
    st.session_state.answer_buffer = ""

if "last_tick" not in st.session_state:
    st.session_state.last_tick = time.time()

# ------------------------------------------------------------
# TITLE BAR
# ------------------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🎈 QuickCalc</h1>", unsafe_allow_html=True)

# ------------------------------------------------------------
# MECHANICS FILE SELECTOR
# ------------------------------------------------------------
st.sidebar.header("Game Setup")
mechanics_path = st.sidebar.text_input(
    "Mechanics CSV Path",
    value="new_item_mechanics.csv"
)

start_button = st.sidebar.button("Start Game")

# ------------------------------------------------------------
# GAME START LOGIC
# ------------------------------------------------------------


if start_button:
    mechanics_df = load_mechanics(mechanics_path)
    st.session_state.engine = QuickCalcEngine(mechanics_df)
    st.session_state.game_running = True
    st.rerun()    # optional — ensures immediate start


# ------------------------------------------------------------
# IF NOT STARTED, REVEAL NOTHING
# ------------------------------------------------------------
if st.session_state.engine is None:
    st.info("Enter the path to your mechanics CSV and press **Start Game**.")
    st.stop()

engine = st.session_state.engine

# ------------------------------------------------------------
# GAME TICK UPDATE
# ------------------------------------------------------------
now = time.time()
delta = now - st.session_state.last_tick   # seconds since last frame
engine.update(delta)                       # UPDATE ENGINE FIRST

# ------------------------------------------------------------
# LAYOUT: 4 CHANNELS + INFO PANEL
# ------------------------------------------------------------
cols = st.columns([1,1,1,1,0.9])

for i, col in enumerate(cols[:4]):
    with col:
        st.markdown(
            f"""
            <div style="position:relative; width:200px; height:300px; border:1px solid #eee; background:#fdfdfd">
            """,
            unsafe_allow_html=True
        )
        # Render balloons in this channel
        for balloon in engine.balloons:
            if balloon.channel == i:
                y_pct = engine.balloon_y_fraction(balloon)
                st.markdown(
                    render_balloon_html(balloon, y_pct),
                    unsafe_allow_html=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# INFO PANEL
# ------------------------------------------------------------
with cols[4]:
    st.markdown("### 🎯 Info")
    st.write(f"**Level:** {engine.level}")
    st.write(f"**Correct:** {engine.correct_count}")
    st.write(f"**Missed:** {engine.missed_count}")

    st.markdown("---")
    st.write("### Input")
    st.text_input("Answer:", key="answer_buffer", value=st.session_state.answer_buffer)

    key_pressed = render_keypad()
    if key_pressed:
        if key_pressed == "⌫":
            st.session_state.answer_buffer = st.session_state.answer_buffer[:-1]
        elif key_pressed == "Enter":
            if st.session_state.answer_buffer.strip():
                engine.submit_answer(st.session_state.answer_buffer.strip())
                st.session_state.answer_buffer = ""
        else:
            st.session_state.answer_buffer += key_pressed

    # Keyboard Enter submission
    if st.session_state.answer_buffer.endswith("\n"):
        cleaned = st.session_state.answer_buffer.strip()
        if cleaned:
            engine.submit_answer(cleaned)
        st.session_state.answer_buffer = ""

    st.markdown("---")
    if st.button("End Game"):
        engine.end_session()
        st.success("Game Ended. Logs saved.")

# ------------------------------------------------------------
# AUTOTICK (10FPS) - must be last
# ------------------------------------------------------------
if st.session_state.game_running:

    # initialise once
    if "last_tick" not in st.session_state:
        st.session_state.last_tick = time.time()

    now = time.time()

    # 10 FPS
    if now - st.session_state.last_tick >= 0.10:
        st.session_state.last_tick = now
        st.rerun()