import html
import streamlit as st
from transformers import pipeline

MODEL_ID = "Yxian909/ner-malaysian"   # your Hugging Face model, or a local folder path
KEEP = {"PER", "ORG", "LOC"}
COLORS = {"PER": "#ffd54f", "ORG": "#81d4fa", "LOC": "#a5d6a7"}


@st.cache_resource
def load_model():
    return pipeline("token-classification", model=MODEL_ID,
                    aggregation_strategy="first")


def detect(ner, text):
    """Run the model, keep PER/ORG/LOC, and join neighbouring entities of the same type."""
    ents = [e for e in ner(text) if e["entity_group"] in KEEP]
    merged = []
    for e in ents:
        if (merged and merged[-1]["entity_group"] == e["entity_group"]
                and e["start"] - merged[-1]["end"] <= 1):
            merged[-1]["end"] = e["end"]
        else:
            merged.append({"entity_group": e["entity_group"],
                           "start": e["start"], "end": e["end"],
                           "score": float(e["score"])})
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


st.set_page_config(page_title="PII Detector", page_icon="🔍")
st.title("Personal Information Detector")
st.caption("Finds people (PER), organisations (ORG) and locations (LOC) "
           "with a fine-tuned DistilBERT model, and redacts them.")

example = "Nur Aisyah binti Ahmad studies at UTM in Johor Bahru."
text = st.text_area("Paste or type text", value=example, height=180)

if st.button("Detect") and text.strip():
    with st.spinner("Loading model and analysing..."):
        ner = load_model()
        ents = detect(ner, text)

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

    st.subheader("Redacted text")
    st.code(redact(text, ents), language=None)
