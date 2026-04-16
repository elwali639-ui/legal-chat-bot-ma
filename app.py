import streamlit as st
import os, json, re, torch
from sentence_transformers import SentenceTransformer, util
import whisper
from gtts import gTTS
import arabic_reshaper
from bidi.algorithm import get_display
import base64
from io import BytesIO

# Configuration page
st.set_page_config(page_title="المساعد القانوني المغربي", page_icon="⚖️", layout="wide")

# Titre
st.title("🎙️ المساعد القانوني المغربي")
st.markdown("### اسأل عن الإجراءات والوثائق القانونية (Darija / Français / Tamazight)")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ الإعدادات")
    use_voice = st.checkbox("🔊 استمع للإجابة", value=True)
    st.info("💡 هذه الأداة للمساعدة الأولية فقط. راجع النصوص الرسمية.")

# Charger les données procédurales
@st.cache_resource
def load_procedures():
    if os.path.exists("procedures_clean.json"):
        with open("procedures_clean.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

procedures = load_procedures()

# Charger modèles (cache)
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
    stt_model = whisper.load_model("base", device=device)
    return embed_model, stt_model, device

# Afficher chargement
if procedures:
    st.success(f"✅ {len(procedures)} صفحات قانونية محملة")
else:
    st.error("❌ لم يتم العثور على procedures_clean.json")
    st.stop()

# Charger modèles
with st.spinner("🧠 جاري تحميل النماذج..."):
    embed_model, stt_model, device = load_models()
    st.success(f"✅ النماذج محملة ({device.upper()})")

# Fonctions
def clean_arabic(text):
    if not text: return ""
    try:
        text = arabic_reshaper.reshape(text)
        text = get_display(text)
    except: pass
    return re.sub(r'\s+', ' ', text).strip()

def search_procedures(query, k=3):
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    texts = [p["text"][:400] for p in procedures]
    doc_emb = embed_model.encode(texts, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, doc_emb)[0]
    top_k = torch.topk(scores, k=min(k, len(scores)))
    
    results = []
    for idx, score in zip(top_k.indices, top_k.values):
        i = idx.item()
        results.append({
            "source": procedures[i].get("source", "?"),
            "page": procedures[i].get("page", "?"),
            "content": clean_arabic(procedures[i]["text"][:450]),
            "score": score.item()
        })
    return results

def text_to_speech(text):
    try:
        lang = "fr" if any('\u00C0' <= c <= '\u024F' for c in text) else "ar"
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

def speech_to_text(audio_file):
    try:
        result = stt_model.transcribe(audio_file, language="ar")
        return result["text"].strip()
    except Exception as e:
        return f"Erreur: {str(e)}"

# Interface chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input utilisateur
col1, col2 = st.columns([3, 1])
with col1:
    text_input = st.text_input("اكتب سؤالك...", key="txt")
with col2:
    audio_input = st.audio_input("🎤", key="aud")

# Traitement
if text_input or audio_input:
    question = text_input
    
    if audio_input:
        with st.spinner("🎧 جاري الاستماع..."):
            question = speech_to_text(audio_input)
            if "Erreur" in question or not question:
                st.error("❌ لم أفهم. حاول مرة أخرى أو اكتب السؤال.")
                question = None
    
    if question:
        # Message utilisateur
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(f"**{question}**")
        
        # Réponse bot
        with st.chat_message("assistant"):
            with st.spinner("🔍 جاري البحث..."):
                results = search_procedures(question, k=3)
                
                if not results:
                    response = "❌ Ma l9itech chi procédure f les documents."
                else:
                    response = f"🔍 **Jawab dyalek:** '{question}'\n\n"
                    for i, r in enumerate(results, 1):
                        response += f"📄 **{r['source']}** (ص.{r['page']})\n"
                        response += f"_{r['content']}..._\n\n"
                        response += "---\n\n"
                    response += "\n💡 *Hadchi gha aide. Rje3 l source officielle.*"
                
                st.markdown(response)
                
                # Audio si demandé
                if use_voice and results:
                    audio_bytes = text_to_speech(response)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mpeg")
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("*⚠️ هذه الأداة للمساعدة الأولية فقط. المرجع الرسمي: الجريدة الرسمية والمحاكم المغربية.*")
