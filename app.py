import streamlit as st
import os, json, re, tempfile
from sentence_transformers import SentenceTransformer, util
import torch
import whisper
from gtts import gTTS
from io import BytesIO

# Configuration
st.set_page_config(page_title="المساعد القانوني المغربي", page_icon="⚖️", layout="wide")

# CSS
st.markdown("""
<style>
    .main-button {
        background-color: #0066cc;
        color: white;
        font-size: 20px;
        padding: 15px 30px;
        border-radius: 10px;
        border: none;
        width: 100%;
    }
    .success-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Charger données
@st.cache_resource
def load_data():
    if os.path.exists("procedures_clean.json"):
        with open("procedures_clean.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# Charger modèles
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    stt_model = whisper.load_model("base")
    return embed_model, stt_model

# Initialisation
procedures = load_data()
if not procedures:
    st.error("❌ procedures_clean.json مفقود!")
    st.stop()

embed_model, stt_model = load_models()

# Interface
st.title("⚖️ المساعد القانوني المغربي")
st.markdown(f"### ✅ {len(procedures)} صفحة قانونية جاهزة")

# Sidebar
with st.sidebar:
    st.markdown("### الإعدادات")
    show_text = st.checkbox("عرض النصوص الأصلية", value=True)
    voice_output = st.checkbox("قراءة الإجابة صوتياً", value=True)

# Input texte
st.markdown("### ✍️ اطرح سؤالك")
question = st.text_input(
    "اكتب سؤالك هنا:",
    placeholder="مثال: ما هي الوثائق المطلوبة للطلاق؟",
    key="question_input"
)

# Input audio
st.markdown("### 🎤 أو سجل صوتك")
audio_file = st.audio_input("سجل سؤالك صوتياً", key="audio_recorder")

# Traitement audio
if audio_file and not question:
    with st.spinner(" جاري تحويل الصوت إلى نص..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_file.getvalue())
                tmp_path = tmp_file.name
            
            result = stt_model.transcribe(tmp_path, language="ar")
            question = result["text"].strip()
            os.unlink(tmp_path)
            
            if question:
                st.success(f"✅ تم فهم: '{question}'")
            else:
                st.error("❌ لم أتمكن من فهم التسجيل. حاول مرة أخرى أو اكتب السؤال.")
        except Exception as e:
            st.error(f"❌ خطأ في التعرف على الصوت: {str(e)}")
            question = ""

# BOTTON ENVOYER
if st.button("🔍 إرسال والبحث", key="search_button", use_container_width=True):
    if not question or len(question.strip()) < 3:
        st.error("⚠️ يرجى كتابة سؤال واضح (3 أحرف على الأقل) أو تسجيل صوتي")
        st.stop()
    
    # Afficher question
    st.markdown(f"<div class='success-box'><b>سؤالك:</b> {question}</div>", unsafe_allow_html=True)
    
    with st.spinner("🔍 جاري البحث في الوثائق القانونية..."):
        try:
            # Recherche sémantique
            query_embedding = embed_model.encode(question, convert_to_tensor=True)
            texts = [p["text"][:400] for p in procedures]
            doc_embeddings = embed_model.encode(texts, convert_to_tensor=True)
            
            # Similarité
            scores = util.cos_sim(query_embedding, doc_embeddings)[0]
            top_k = torch.topk(scores, k=min(3, len(scores)))
            
            if top_k.values[0].item() < 0.3:
                st.warning("⚠️ لم أجد معلومات دقيقة عن هذا الموضوع. إليك أقرب النتائج:")
            
            # Afficher résultats
            response_text = f" **النتائج المتعلقة بـ '{question}':**\n\n"
            
            for idx, score in zip(top_k.indices, top_k.values):
                i = idx.item()
                doc = procedures[i]
                response_text += f"📄 **المصدر:** {doc.get('source', 'غير معروف')} (صفحة {doc.get('page', '?')})\n"
                response_text += f"📝 **النص:** {doc['text'][:300]}...\n\n"
                response_text += "---\n\n"
            
            response_text += "\n💡 *ملاحظة: هذه الأداة للمساعدة الأولية فقط. راجع دائماً المصادر الرسمية.*"
            
            # Afficher
            st.markdown(response_text.replace("\n", "<br>"), unsafe_allow_html=True)
            
            # Audio si demandé
            if voice_output:
                with st.spinner("🔊 جاري تحويل النص إلى صوت..."):
                    try:
                        lang = "fr" if any('\u00C0' <= c <= '\u024F' for c in response_text) else "ar"
                        tts = gTTS(text=response_text, lang=lang, slow=False)
                        audio_bytes = BytesIO()
                        tts.write_to_fp(audio_bytes)
                        audio_bytes.seek(0)
                        st.audio(audio_bytes, format="audio/mpeg")
                    except Exception as e:
                        st.warning(f"⚠️ مشكلة في الصوت: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ خطأ في البحث: {str(e)}")
            st.error("💡 حاول سؤالاً آخر أو تحقق من اتصالك بالإنترنت")

# Footer
st.markdown("---")
st.caption("⚖️ المساعد القانوني المغربي - للمساعدة الأولية فقط")
