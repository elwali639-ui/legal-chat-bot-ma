import streamlit as st
import os, json, tempfile, torch
from sentence_transformers import SentenceTransformer, util
from faster_whisper import WhisperModel
from gtts import gTTS
from io import BytesIO
import google.generativeai as genai

# ================= CONFIG =================
st.set_page_config(page_title="المساعد القانوني المغربي الذكي", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;700&display=swap');
    .stApp { font-family: 'Noto Naskh Arabic', serif; }
    .main-title { text-align: center; color: #1a5276; font-size: 2.2em; font-weight: bold; margin-bottom: 10px; }
    .search-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin: 15px 0; }
    .result-card { background: white; border-left: 5px solid #667eea; padding: 20px; border-radius: 10px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .stButton>button { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; font-size: 18px; font-weight: bold; padding: 12px 0; border-radius: 25px; border: none; width: 100%; }
    .ai-badge { background: #e8f5e9; color: #2e7d32; padding: 5px 10px; border-radius: 5px; font-size: 0.9em; display: inline-block; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ================= CHARGEMENT =================
@st.cache_resource
def load_resources():
    if os.path.exists("procedures_clean.json"):
        with open("procedures_clean.json", "r", encoding="utf-8") as f:
            procedures = json.load(f)
    else:
        return None, None, None
    
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return procedures, embed_model, stt_model

procedures, embed_model, stt_model = load_resources()
if not procedures:
    st.error("❌ ملف procedures_clean.json غير موجود!"); st.stop()

# ================= FONCTIONS =================
def stt_transcribe(audio_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        segments, _ = stt_model.transcribe(tmp_path, language="ar", beam_size=1)
        text = " ".join([seg.text for seg in segments])
        os.unlink(tmp_path)
        return text.strip()
    except Exception as e:
        return f"خطأ في الصوت: {str(e)}"

def search_context(query, k=3):
    q_emb = embed_model.encode(query, convert_to_tensor=True)
    texts = [p["text"][:400] for p in procedures]
    t_emb = embed_model.encode(texts, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, t_emb)[0]
    top_k = torch.topk(scores, k=min(k, len(scores)))
    
    context = ""
    for idx in top_k.indices:
        i = idx.item()
        context += f"📄 [{procedures[i].get('source','?')} ص.{procedures[i].get('page','?')}]\n{procedures[i]['text'][:300]}\n\n"
    return context, scores[top_k.indices[0]].item()

def get_ai_answer(question, context, api_key):
    if not api_key or len(api_key.strip()) < 10:
        return None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = f"""أنت خبير قانوني مغربي. أجب على سؤال المستخدم باللغة العربية الفصحى المبسطة أو الدارجة المغربية الواضحة، بناءً حصراً على السياق القانوني التالي.
إذا لم تكن المعلومة موجودة في السياق، اذكر ذلك بوضوح.
كن دقيقاً، مختصراً، ومنظماً في نقاط إذا لزم الأمر.

📜 السياق القانوني:
{context}

❓ سؤال المستخدم:
{question}

✅ الإجابة:"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ خطأ في الذكاء الاصطناعي: {str(e)}"

def text_to_speech(text):
    try:
        lang = "fr" if any('\u00C0' <= c <= '\u024F' for c in text) else "ar"
        tts = gTTS(text=text, lang=lang, slow=False)
        buf = BytesIO(); tts.write_to_fp(buf); buf.seek(0)
        return buf
    except: return None

# ================= INTERFACE =================
st.markdown('<div class="main-title">⚖️ المساعد القانوني المغربي الذكي</div>', unsafe_allow_html=True)
st.markdown('<span class="ai-badge">🤖 مدعوم بـ Google Gemini AI (مجاني)</span>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ الإعدادات")
    gemini_key = st.text_input("🔑 مفتاح Google Gemini API", type="password", 
                               help="احصل عليه مجاناً من aistudio.google.com/apikey")
    voice_out = st.checkbox("🔊 قراءة الإجابة صوتياً", value=True)
    st.info("💡 بدون المفتاح: بحث مباشر في النصوص. مع المفتاح: إجابات ذكية بالدارجة/العربية.")

st.markdown('<div class="search-box">', unsafe_allow_html=True)
question = st.text_input("✍️ اكتب سؤالك:", placeholder="مثال: شنو هي الوثائق ديال الطلاق؟ / Comment créer une SARL؟", label_visibility="collapsed")
audio_file = st.audio_input("🎤 أو سجل صوتك هنا", key="voice_rec")
st.markdown('</div>', unsafe_allow_html=True)

if audio_file and not question:
    with st.spinner("🎧 جاري تحويل الصوت إلى نص..."):
        question = stt_transcribe(audio_file.getvalue())
        if question and len(question) > 3:
            st.success(f"✅ تم الفهم: '{question}'")
        else:
            st.error("❌ لم أفهم التسجيل. حاول مرة أخرى أو اكتب السؤال.")
            question = ""

if st.button("🔍 بحث ذكي مع AI", key="search_btn"):
    if not question or len(question.strip()) < 3:
        st.warning("⚠️ يرجى كتابة سؤال أو تسجيل صوتي."); st.stop()
    
    with st.spinner("🔍 جاري البحث وتوليد الإجابة..."):
        context, score = search_context(question, k=3)
        
        final_answer = ""
        if gemini_key and len(gemini_key.strip()) > 10:
            final_answer = get_ai_answer(question, context, gemini_key)
            if not final_answer or "خطأ" in final_answer:
                final_answer = f"📋 **النتائج المباشرة:**\n\n{context}"
        else:
            final_answer = f"📋 **النتائج المتعلقة بـ '{question}':**\n\n{context}"
        
        st.markdown(f'<div class="result-card">{final_answer.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        
        if voice_out:
            with st.spinner("🔊 جاري القراءة..."):
                audio = text_to_speech(final_answer)
                if audio: st.audio(audio, format="audio/mpeg")

st.markdown("---")
st.caption("⚖️ للمساعدة الأولية فقط. المرجع الرسمي: الجريدة الرسمية والمحاكم المغربية.")
