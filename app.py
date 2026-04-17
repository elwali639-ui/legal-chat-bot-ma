import streamlit as st
import os, json, re, requests, base64
from sentence_transformers import SentenceTransformer, util
import torch
from gtts import gTTS
from io import BytesIO

# Configuration
st.set_page_config(
    page_title="المساعد القانوني المغربي الذكي",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Moderne
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;700&display=swap');
    
    .stApp { font-family: 'Noto Naskh Arabic', serif; }
    
    .main-title {
        text-align: center;
        color: #1a5276;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    .search-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
    }
    
    .result-card {
        background: white;
        border-left: 5px solid #667eea;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 15px 40px;
        border-radius: 25px;
        border: none;
        width: 100%;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ================= CHARGEMENT =================
@st.cache_resource
def load_resources():
    # Données
    if os.path.exists("procedures_clean.json"):
        with open("procedures_clean.json", "r", encoding="utf-8") as f:
            procedures = json.load(f)
    else:
        return None, None, None
    
    # Modèles
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    return procedures, embed_model, None

procedures, embed_model, _ = load_resources()

if not procedures:
    st.error("❌ ملف procedures_clean.json غير موجود!")
    st.stop()

# ================= INTERFACE =================
st.markdown('<div class="main-title">⚖️ المساعد القانوني المغربي الذكي</div>', unsafe_allow_html=True)
st.markdown(f"### ✅ {len(procedures)} وثيقة قانونية جاهزة للبحث")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ الإعدادات المتقدمة")
    
    # API Configuration
    use_ai = st.checkbox("🤖 استخدام الذكاء الاصطناعي (أفضل)", value=False)
    hf_token = ""
    if use_ai:
        hf_token = st.text_input("مفتاح Hugging Face API", type="password", 
                                  help="احصل عليه مجاناً من huggingface.co/settings/tokens")
    
    voice_output = st.checkbox("🔊 تشغيل الإجابة صوتياً", value=True)
    show_sources = st.checkbox("📚 عرض المصادر الأصلية", value=True)
    
    st.markdown("---")
    st.info("💡 **نصيحة:** بدون مفتاح API = بحث مباشر في الوثائق. مع المفتاح = إجابات ذكية مفصلة.")

# ================= FONCTIONS =================
def search_documents(query, k=3):
    """Recherche sémantique avancée"""
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    texts = [p["text"][:500] for p in procedures]
    doc_emb = embed_model.encode(texts, convert_to_tensor=True)
    
    scores = util.cos_sim(query_emb, doc_emb)[0]
    top_k = torch.topk(scores, k=min(k, len(scores)))
    
    results = []
    for idx, score in zip(top_k.indices, top_k.values):
        i = idx.item()
        results.append({
            "source": procedures[i].get("source", "غير معروف"),
            "page": procedures[i].get("page", "?"),
            "content": procedures[i]["text"],
            "score": score.item()
        })
    return results

def generate_ai_response(question, context, api_key=""):
    """Génération de réponse avec LLM"""
    if not api_key:
        return None
    
    prompt = f"""أنت مساعد قانوني مغربي خبير. أجب على السؤال بناءً على السياق التالي فقط.
استخدم لغة عربية قانونية واضحة أو الدارجة المغربية المفهومة.
كن دقيقاً ومختصراً.

السياق القانوني:
{context}

السؤال: {question}

الإجابة:"""

    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 600,
            "temperature": 0.3
        }
        
        response = requests.post(
            "https://api-inference.huggingface.co/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
    except:
        pass
    
    return None

def text_to_speech(text):
    """TTS sans FFmpeg"""
    try:
        lang = "fr" if any('\u00C0' <= c <= '\u024F' for c in text) else "ar"
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"❌ خطأ في الصوت: {str(e)}")
        return None

# ================= ZONE DE RECHERCHE =================
st.markdown('<div class="search-box">', unsafe_allow_html=True)
st.markdown("### 🔍 اطرح سؤالك القانوني")

# Input texte
question = st.text_input(
    "اكتب سؤالك هنا:",
    placeholder="مثال: ما هي الوثائق المطلوبة لتسجيل شركة؟ / شنو هي إجراءات الطلاق؟",
    label_visibility="collapsed",
    key="main_question"
)

# Bouton recherche
col1, col2 = st.columns([3, 1])
with col1:
    search_clicked = st.button("🔍 بحث ذكي", key="search_btn", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ================= TRAITEMENT =================
if search_clicked and question and len(question.strip()) > 3:
    with st.spinner("🔍 جاري البحث في الوثائق القانونية..."):
        results = search_documents(question, k=3)
        
        if not results or results[0]["score"] < 0.3:
            st.warning("⚠️ لم أجد معلومات دقيقة. إليك أقرب النتائج المتاحة:")
        
        # Contexte pour AI
        context_text = "\n\n".join([f"[{r['source']} ص.{r['page']}]: {r['content'][:300]}" for r in results])
        
        # Génération réponse
        final_response = ""
        
        if use_ai and hf_token:
            with st.spinner("🤖 الذكاء الاصطناعي يصيغ الإجابة..."):
                ai_response = generate_ai_response(question, context_text, hf_token)
                if ai_response:
                    final_response = ai_response
                else:
                    final_response = f"📋 **بناءً على الوثائق المتاحة:**\n\n{context_text}"
        else:
            final_response = f"📋 **النتائج المتعلقة بـ '{question}':**\n\n"
            for i, r in enumerate(results, 1):
                final_response += f"**{i}. {r['source']}** (صفحة {r['page']}) - درجة المطابقة: {r['score']:.1%}\n"
                final_response += f"{r['content'][:250]}...\n\n"
                final_response += "---\n\n"
        
        # Afficher réponse
        st.markdown(f'<div class="result-card">{final_response.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        
        # Audio
        if voice_output:
            with st.spinner("🔊 جاري قراءة الإجابة..."):
                audio = text_to_speech(final_response)
                if audio:
                    st.audio(audio, format="audio/mpeg")
        
        # Sources détaillées
        if show_sources:
            with st.expander("📚 عرض الوثائق الأصلية الكاملة"):
                for r in results:
                    st.markdown(f"**{r['source']}** (صفحة {r['page']})")
                    st.text(r['content'])
                    st.markdown("---")

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>⚖️ <b>تنبيه مهم:</b> هذه الأداة للمساعدة الأولية فقط. المرجع الرسمي هو الجريدة الرسمية والمحاكم المغربية.</p>
    <p>📞 للاستفسارات: وزارة العدل - www.justice.gov.ma</p>
</div>
""", unsafe_allow_html=True)

# ================= EXEMPLES =================
with st.expander("💡 أمثلة على الأسئلة"):
    st.markdown("""
    - ما هي الوثائق المطلوبة لتسجيل شركة ذات مسؤولية محدودة؟
    - ما هي إجراءات الطلاق بالمغرب؟
    - كيف يمكن الحصول على شهادة السوابق العدلية؟
    - ما هي شروط التسجيل في السجل التجاري؟
    - كم تبلغ رسوم الطلاق؟
    """)
