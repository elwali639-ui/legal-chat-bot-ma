import streamlit as st
import os, json, torch, requests, tempfile, re
from sentence_transformers import SentenceTransformer, util
import whisper
from gtts import gTTS
from io import BytesIO
import arabic_reshaper
from bidi.algorithm import get_display

# ================= CONFIGURATION =================
st.set_page_config(page_title="المساعد القانوني المغربي", page_icon="⚖️", layout="wide")

# Style CSS
st.markdown("""
<style>
    .stButton>button {
        background: linear-gradient(90deg, #0066cc 0%, #004499 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px 0;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background: linear-gradient(90deg, #0077ee 0%, #0055bb 100%); }
    .result-box { background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #0066cc; }
    .error-box { background: #ffe6e6; padding: 10px; border-radius: 5px; border-left: 4px solid #cc0000; }
</style>
""", unsafe_allow_html=True)

# ================= CHARGEMENT DES DONNÉES & MODÈLES =================
@st.cache_resource
def load_resources():
    # 1. Données JSON
    if os.path.exists("procedures_clean.json"):
        with open("procedures_clean.json", "r", encoding="utf-8") as f:
            procedures = json.load(f)
    else:
        st.error("❌ ملف procedures_clean.json مفقود!"); st.stop()
    
    # 2. Modèles Embedding & Whisper (CPU optimisé)
    device = "cpu"
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
    stt_model = whisper.load_model("base", device=device)
    return procedures, embed_model, stt_model

procedures, embed_model, stt_model = load_resources()
st.success(f"✅ {len(procedures)} صفحة قانونية جاهزة | 🧠 النماذج محملة")

# ================= FONCTIONS UTILITAIRES =================
def clean_arabic(text):
    if not text: return ""
    try:
        text = arabic_reshaper.reshape(text)
        text = get_display(text)
    except: pass
    return re.sub(r'\s+', ' ', text).strip()

def search_rag(query, k=3):
    """Recherche sémantique locale"""
    q_emb = embed_model.encode(query, convert_to_tensor=True)
    texts = [clean_arabic(p["text"][:400]) for p in procedures]
    t_emb = embed_model.encode(texts, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, t_emb)[0]
    top_k = torch.topk(scores, k=min(k, len(scores)))
    
    context = ""
    for idx in top_k.indices:
        i = idx.item()
        context += f"[{procedures[i].get('source','?')} ص.{procedures[i].get('page','?')}]: {procedures[i]['text'][:350]}\n\n"
    return context, scores[top_k.indices[0]].item()

def get_llm_response(context, question, api_key=""):
    """Synthèse intelligente via API (HF Gratuit ou OpenAI)"""
    if not api_key:
        return None  # Fallback local
    
    # Prompt optimisé pour droit marocain + Darija/Arabe
    prompt = f"""أنت خبير قانوني مغربي. أجب على السؤال بناءً حصراً على السياق التالي.
إذا لم تجد الإجابة، قل بصراحة: 'لا تتوفر معلومات كافية في الوثائق المتاحة'.
استخدم لغة عربية قانونية واضحة أو الدارجة المغربية المفهومة.
السياق:
{context}
السؤال: {question}
الإجابة:"""

    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.3
        }
        res = requests.post("https://api-inference.huggingface.co/v1/chat/completions", headers=headers, json=payload)
        return res.json()["choices"][0]["message"]["content"]
    except:
        return None

def audio_to_text(audio_bytes):
    """STT Whisper"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        result = stt_model.transcribe(tmp_path, language="ar")
        os.unlink(tmp_path)
        return result["text"].strip()
    except: return ""

def text_to_audio(text):
    """TTS gTTS"""
    try:
        lang = "fr" if any('\u00C0' <= c <= '\u024F' for c in text) else "ar"
        tts = gTTS(text=text, lang=lang, slow=False)
        buf = BytesIO(); tts.write_to_fp(buf); buf.seek(0)
        return buf
    except: return None

# ================= INTERFACE UTILISATEUR =================
st.title("🎙️ المساعد القانوني المغربي الذكي")
st.markdown("اسأل عن الإجراءات، الوثائق، المساطر... (Darija / Français / Tamazight)")

with st.sidebar:
    st.markdown("### ⚙️ إعدادات متقدمة")
    hf_token = st.text_input("🔑 مفتاح Hugging Face API (اختياري للجودة العالية)", type="password", help="احصل عليه مجاناً من huggingface.co/settings/tokens")
    use_voice_out = st.checkbox("🔊 قراءة الإجابة صوتياً", value=True)
    st.info("💡 بدون مفتاح API: إجابات مباشرة من الوثائق. مع المفتاح: ذكاء اصطناعي متقدم.")

# Inputs
col1, col2 = st.columns([3, 1])
with col1:
    user_text = st.text_input("✍️ اكتب سؤالك هنا...", placeholder="مثال: شنو هي الوثائق اللي خاصني للطلاق؟", key="txt_in")
with col2:
    user_audio = st.audio_input("🎤 سجل صوتك", key="aud_in")

# BOTTON ENVOYER CLAIR
if st.button("🚀 إرسال والبحث", key="send_btn"):
    question = user_text.strip()
    
    # Traitement Audio si présent
    if user_audio and not question:
        with st.spinner("🎧 جاري تحويل الصوت لنص..."):
            question = audio_to_text(user_audio)
            if not question:
                st.error("❌ لم أفهم التسجيل. حاول مرة أخرى أو اكتب السؤال."); st.stop()
            st.info(f"🗣️ تم فهم: '{question}'")
    
    if not question:
        st.warning("⚠️ يرجى كتابة سؤال أو تسجيل صوتي"); st.stop()
    
    # Affichage question
    st.markdown(f"<div class='result-box'><b>❓ سؤالك:</b> {question}</div>", unsafe_allow_html=True)
    
    with st.spinner("🔍 جاري البحث في الوثائق القانونية..."):
        context, score = search_rag(question, k=3)
        
        if score < 0.5:
            st.warning("⚠️没有找到完全匹配的信息。عرض أقرب النتائج المتاحة.")
        
        # Synthèse LLM si clé fournie, sinon affichage direct
        final_answer = ""
        if hf_token:
            with st.spinner("🤖 الذكاء الاصطناعي يصيغ الإجابة..."):
                final_answer = get_llm_response(context, question, hf_token)
        
        if not final_answer:
            # Fallback local propre
            final_answer = f"📖 **بناءً على الوثائق المتاحة:**\n\n{context}\n💡 *ملاحظة: هذه النتائج مستخرجة آلياً من النصوص القانونية. يرجى التأكد من المصادر الرسمية.*"
        
        # Affichage réponse
        st.markdown(f"<div class='result-box'><b>✅ الإجابة:</b><br>{final_answer.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
        
        # Audio réponse
        if use_voice_out:
            audio_buf = text_to_audio(final_answer)
            if audio_buf:
                st.audio(audio_buf, format="audio/mpeg", start_time=0)

# Footer
st.markdown("---")
st.caption("⚖️ هذه الأداة للمساعدة الأولية فقط. المرجع الرسمي: الجريدة الرسمية والمحاكم المغربية.")
