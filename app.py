import html
import re
import streamlit as st
from pypdf import PdfReader
from transformers import pipeline

MODEL_ID = "Yxian909/ner-malaysian"
KEEP = {"PER", "ORG", "LOC"}
COLORS = {"PER": "#ffd54f", "ORG": "#81d4fa", "LOC": "#a5d6a7"}
MAX_CHARS = 100_000   # keeps long documents fast enough on a laptop
MAX_PAGES = 30


@st.cache_resource
def load_model():
    # stride lets the pipeline process texts longer than the model's 512-token limit
    return pipeline("token-classification", model=MODEL_ID,
                    aggregation_strategy="first", stride=32)


def detect(ner, text):
    ABBREV = {"dr", "prof", "encik", "puan", "cik", "datuk", "tun", "sdn", "bhd", "mr", "mrs", "ms"}
    
    def split_sentences(text):
        """(start, end) of sentence-like chunks; offsets refer to the original text."""
        spans, start = [], 0
        for m in re.finditer(r"[.!?]+\s+|\n+", text):
            before = text[start:m.start()].split()
            last = before[-1].lower().rstrip(".") if before else ""
            if m.group().strip() and last in ABBREV:
                continue
            if text[start:m.end()].strip():
                spans.append((start, m.end()))
            start = m.end()
        if text[start:].strip():
            spans.append((start, len(text)))
        return spans
    
    def prep(chunk):
        """Title-case ALL-CAPS headings for the model only (same length, so offsets stay valid)."""
        letters = [c for c in chunk if c.isalpha()]
        if len(letters) >= 12 and all(c.isupper() for c in letters):
            return chunk.title()
        return chunk
    
    def detect(ner, text):
        ents = []
        for s, e in split_sentences(text):
            for x in ner(prep(text[s:e])):
                if x["entity_group"] in KEEP:
                    ents.append({"entity_group": x["entity_group"],
                                 "start": x["start"] + s, "end": x["end"] + s,
                                 "score": float(x["score"])})
        # join entities separated only by a space (fixes "Pulau" + "Pinang" splits)
        merged = []
        for e in sorted(ents, key=lambda z: z["start"]):
            if merged and text[merged[-1]["end"]:e["start"]].strip() == "" \
                    and e["start"] - merged[-1]["end"] <= 1:
                prev = merged[-1]
                if e["score"] > prev["score"]:
                    prev["entity_group"] = e["entity_group"]
                prev["end"], prev["score"] = e["end"], max(prev["score"], e["score"])
            else:
                merged.append(dict(e))
        for m in merged:
            m["word"] = text[m["start"]:m["end"]]
        return merged


def redact(text, ents):
    """Replace each detected entity with its type label, e.g. [PER]."""
    for e in sorted(ents, key=lambda x: x["start"], reverse=True):
        text = text[:e["start"]] + f"[{e['entity_group']}]" + text[e["end"]:]
    return text


def highlight(text, ents):
    """Return HTML with each entity highlighted and labelled."""
    out, pos = "", 0
    for e in sorted(ents, key=lambda x: x["start"]):
        out += html.escape(text[pos:e["start"]])
        out += (f'<mark style="background:{COLORS[e["entity_group"]]};'
                f'padding:2px 4px;border-radius:4px;color:#000">'
                f'{html.escape(text[e["start"]:e["end"]])} '
                f'<small><b>{e["entity_group"]}</b></small></mark>')
        pos = e["end"]
    out += html.escape(text[pos:])
    return out.replace("\n", "<br>")


def read_file(uploaded):
    """Extract plain text from a .txt or .pdf upload (extraction only; detection is our own model)."""
    name = uploaded.name.lower()
    if name.endswith(".txt"):
        return uploaded.read().decode("utf-8", errors="replace")
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded)
        pages = reader.pages[:MAX_PAGES]
        return "\n".join((p.extract_text() or "") for p in pages)
    return ""


def analyse(text):
    if len(text) > MAX_CHARS:
        st.warning(f"The text is long, so only the first {MAX_CHARS:,} characters are analysed.")
        text = text[:MAX_CHARS]
    with st.spinner("Loading model and analysing..."):
        ner = load_model()
        ents = detect(ner, text)
    st.session_state["result"] = {"text": text, "ents": ents}


st.set_page_config(page_title="PII Detector", page_icon="🔍")
st.title("Personal Information Detector")
st.caption("Finds people (PER), organisations (ORG) and locations (LOC) "
           "with a fine-tuned DistilBERT model, and redacts them. "
           "Uploaded files are processed in memory and are not saved.")

tab_text, tab_file = st.tabs(["Type or paste text", "Upload a file (.txt / .pdf)"])

with tab_text:
    typed = st.text_area("Paste or type text", height=180,
                         value="Nur Aisyah binti Ahmad studies at UTM in Johor Bahru.")
    if st.button("Detect", key="detect_text") and typed.strip():
        analyse(typed)

with tab_file:
    uploaded = st.file_uploader("Choose a .txt or .pdf file", type=["txt", "pdf"])
    if st.button("Detect in file", key="detect_file") and uploaded is not None:
        try:
            file_text = read_file(uploaded)
        except Exception as err:
            file_text = ""
            st.error(f"Could not read this file: {err}")
        if file_text.strip():
            analyse(file_text)
        elif not file_text:
            pass
        else:
            st.warning("No text found in this file. Scanned PDFs (images) need OCR, which this app does not do.")

result = st.session_state.get("result")
if result:
    text, ents = result["text"], result["ents"]
    st.subheader("Detected entities")
    if ents:
        st.markdown(highlight(text, ents), unsafe_allow_html=True)
        st.dataframe(
            [{"Text": e["word"], "Type": e["entity_group"],
              "Confidence": round(e["score"], 3)} for e in ents],
            use_container_width=True,
        )
    else:
        st.info("No people, organisations or locations found.")

    redacted = redact(text, ents)
    st.subheader("Redacted text")
    st.code(redacted, language=None)
    st.download_button("Download redacted text (.txt)", data=redacted,
                       file_name="redacted.txt", mime="text/plain")
