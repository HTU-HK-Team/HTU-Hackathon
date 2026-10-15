"""Local chat UI for the government-services bot (same engine as main.py).

    ..\\.venv\\Scripts\\python.exe -m streamlit run chat_app.py

Full conversation with follow-ups (the chat remembers which service you are talking about),
and every answer shows the exact lines from the official documents it was based on.
"""
import json
import os
import time

import streamlit as st

import main as M

st.set_page_config(page_title="مساعد الأحوال المدنية والجوازات", page_icon="🪪", layout="centered")
st.markdown("""
<style>
  .stChatMessage, .stMarkdown, .stChatInput textarea, .stExpander, .src {direction: rtl; text-align: right;}
  .src {font-size: 0.92rem; background: rgba(128,128,128,0.08); border-right: 3px solid #4a90d9;
        padding: 6px 10px; margin: 4px 0; border-radius: 4px;}
  .tag {display:inline-block; font-size:0.78rem; padding:1px 8px; margin:2px; border-radius:10px;
        background: rgba(74,144,217,0.15);}
  .guard {background: rgba(31,111,235,0.12); border: 1px solid rgba(31,111,235,0.35);}
  /* no red anywhere: chat avatars, input focus, toggles */
  [data-testid="stChatMessageAvatarUser"] {background-color: #1F6FEB !important;}
  [data-testid="stChatMessageAvatarAssistant"] {background-color: #0E8A6E !important;}
  textarea:focus, [data-baseweb="textarea"]:focus-within {border-color: #1F6FEB !important;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def boot():
    services = M.load_services()
    M.ensure_ollama()
    return services


SERVICES = boot()

if "turns" not in st.session_state:
    st.session_state.turns = []        # [{q, a, sources, sids, mode, guards, sec}]
    st.session_state.state = {}        # routing memory for follow-ups (which document / situation)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ⚙️ الإعدادات")
    st.caption(f"Model: `{M.MODEL}`")
    st.caption("Config: " + ", ".join(f"{k}={v}" for k, v in M.CONFIG.items()))
    show_debug = st.toggle("إظهار تفاصيل التوجيه والحماية", value=True)
    if st.button("🔄 محادثة جديدة", use_container_width=True):
        st.session_state.turns, st.session_state.state = [], {}
        st.rerun()
    st.markdown("---")
    st.markdown("**أمثلة للتجربة**")
    EXAMPLES = ["عمري 17 وجوازي بايق عليه 3 شهور، كم رسوم التجديد؟", "وين بقدر أقدّم الطلب؟",
                "الباسبور ضاع، شو بعمل؟", "وكم بتكلف إذا ضاع للمرة الثانية؟",
                "بدي أجدد، شو المطلوب؟", "أيهما أغلى: جواز بدل تالف ولا بدل فاقد؟",
                "إذا توفي الأب والأم، مين بصير رب الأسرة؟", "كم رسوم تصديق الشهادة الجامعية؟"]
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state.pending = ex
            st.rerun()

st.markdown("## 🪪 مساعد دائرة الأحوال المدنية والجوازات")
st.caption("أسئلة عن جواز السفر، دفتر العائلة، والبطاقة الذكية — الإجابات من الوثائق الرسمية فقط، مع ذكر المصدر.")

MODE_LABEL = {"answer": "إجابة", "clarify": "طلب توضيح", "refuse": "خارج النطاق", "error": "خطأ"}


def render_turn(t):
    with st.chat_message("user"):
        st.markdown(t["q"])
    with st.chat_message("assistant"):
        st.markdown(t["a"])
        if show_debug:
            tags = [f'<span class="tag">{MODE_LABEL.get(t["mode"], t["mode"])}</span>']
            tags += [f'<span class="tag">{M.TITLES.get(s, s)}</span>' for s in t["sids"]]
            tags += [f'<span class="tag guard">🛡️ {g}</span>' for g in t["guards"]]
            tags.append(f'<span class="tag">{t["sec"]:.1f}s</span>')
            st.markdown(" ".join(tags), unsafe_allow_html=True)
        if t["sources"]:
            with st.expander(f"📄 المصادر ({len(t['sources'])})"):
                for src in t["sources"]:
                    st.markdown(f'<div class="src">{src}</div>', unsafe_allow_html=True)


for t in st.session_state.turns:
    render_turn(t)

question = st.chat_input("اكتب سؤالك هنا... (مثال: قديش رسوم تجديد الهوية؟)")
if st.session_state.get("pending"):
    question = st.session_state.pop("pending")

if question:
    with st.chat_message("user"):
        st.markdown(question)
    history = [(t["q"], t["a"]) for t in st.session_state.turns]
    before = dict(M.STATS)
    t0 = time.time()
    with st.chat_message("assistant"):
        with st.spinner("جاري البحث في الوثائق..."):
            try:
                ans, ctxs, new_state, mode, sids = M.answer_message(question, history, st.session_state.state, SERVICES)
            except Exception as e:
                ans, ctxs, new_state, mode, sids = f"حدث خطأ: {e}", [], st.session_state.state, "error", []
    guards = []
    d = {k: M.STATS[k] - before.get(k, 0) for k in M.STATS if isinstance(M.STATS[k], int)}
    if d.get("fee_rule_in_code"):
        guards.append("قاعدة الرسوم اختيرت برمجياً")
    if d.get("number_check_regenerated"):
        guards.append("فحص الأرقام: أُعيد التوليد")
    if d.get("llm_fallback"):
        guards.append("تصنيف احتياطي بالنموذج")
    st.session_state.state = new_state            # same as main.py: each answer updates the chat memory
    st.session_state.turns.append({"q": question, "a": ans or "—", "sources": ctxs, "sids": sids, "mode": mode,
                                   "guards": guards, "sec": time.time() - t0})
    st.rerun()
