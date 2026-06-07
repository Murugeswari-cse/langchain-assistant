import os,re,smtplib,streamlit as st, webbrowser, io



from email.message import EmailMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
import pypdf

load_dotenv()

def parse_pdf(file_bytes):
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {e}"

def parse_text(file_bytes):
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return file_bytes.decode("latin1")
        except Exception as e:
            return f"Error reading text file: {e}"

def get_relevant_chunks(text, query, top_k=5, chunk_size=1500, overlap=300):
    # Split by sentence or newline boundaries
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Start next chunk with overlap sentences
            overlap_len = 0
            overlap_sentences = []
            for s in reversed(current_chunk):
                if overlap_len + len(s) < overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s) + 1
                else:
                    break
            current_chunk = overlap_sentences
            current_length = overlap_len
            
        current_chunk.append(sentence)
        current_length += len(sentence) + 1
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    query_words = set(re.findall(r'\w+', query.lower()))
    if not query_words:
        return chunks[:top_k]
        
    scored_chunks = []
    for idx, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = sum(chunk_lower.count(word) for word in query_words)
        scored_chunks.append((score, idx, chunk))
        
    scored_chunks.sort(key=lambda x: (-x[0], x[1]))
    return [c[2] for c in scored_chunks[:top_k]]

st.set_page_config(page_title="AI Pro Assistant", page_icon="⚡", layout="wide")

st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;500&display=swap');

/* Base animated dark theme */
.stApp {
    background: linear-gradient(45deg, #0f172a, #020617, #1e1b4b, #0f172a);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Typography */
h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    text-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
}

/* Chat Input Styling */
.stChatInputContainer {
    border: 2px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 20px !important;
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.2);
    transition: all 0.3s ease;
    background: rgba(15, 23, 42, 0.8) !important;
    backdrop-filter: blur(10px);
}

.stChatInputContainer:focus-within {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 30px rgba(56, 189, 248, 0.6);
    transform: scale(1.02);
}

/* Buttons */
.stButton>button {
    width: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    color: white;
    border: none;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.4s ease;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
    position: relative;
    overflow: hidden;
}

.stButton>button:hover {
    transform: translateY(-3px) scale(1.05);
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6);
}

/* Animated chat messages */
.stChatMessage {
    animation: slideUpFade 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 1rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(5px);
}

@keyframes slideUpFade {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: rgba(2, 6, 23, 0.7) !important;
    backdrop-filter: blur(15px);
    border-right: 1px solid rgba(56, 189, 248, 0.2);
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.2); }
::-webkit-scrollbar-thumb { background: rgba(56, 189, 248, 0.5); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(56, 189, 248, 0.8); }

</style>
""", unsafe_allow_html=True)

st.title("⚡ AI Pro Assistant")
st.caption("Groq + Tavily + Streaming")

primary_llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# Set up fallback models to run if primary model is rate-limited
fallbacks = []

google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        fallbacks.append(ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
            temperature=0
        ))
    except Exception:
        pass

groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    # Use smaller Groq models which have separate rate limit pools
    fallbacks.append(ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        temperature=0
    ))
    fallbacks.append(ChatGroq(
        model="gemma2-9b-it",
        api_key=groq_api_key,
        temperature=0
    ))

if fallbacks:
    llm = primary_llm.with_fallbacks(fallbacks)
else:
    llm = primary_llm

search=TavilySearchResults(
    max_results=5,
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)

if "messages" not in st.session_state:
    st.session_state.messages=[]
if "report" not in st.session_state:
    st.session_state.report=None
if "show_details" not in st.session_state:
    st.session_state.show_details=False
if "uploaded_doc_text" not in st.session_state:
    st.session_state.uploaded_doc_text = None
if "uploaded_doc_name" not in st.session_state:
    st.session_state.uploaded_doc_name = None
if "assistant_mode" not in st.session_state:
    st.session_state.assistant_mode = "Web Search (Tavily)"

# Sidebar UI
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a Text or PDF document to query it directly",
        type=["pdf", "txt", "md"]
    )
    
    # Process uploaded file
    if uploaded_file:
        file_name = uploaded_file.name
        # Check if the file has changed or hasn't been parsed yet
        if st.session_state.uploaded_doc_name != file_name:
            with st.spinner("Processing document..."):
                file_bytes = uploaded_file.read()
                if file_name.endswith(".pdf"):
                    doc_text = parse_pdf(file_bytes)
                else:
                    doc_text = parse_text(file_bytes)
                
                if doc_text.startswith("Error"):
                    st.error(doc_text)
                else:
                    st.session_state.uploaded_doc_text = doc_text
                    st.session_state.uploaded_doc_name = file_name
                    st.session_state.assistant_mode = "Ask Uploaded Document"
                    st.success(f"Successfully loaded {file_name}!")
                    st.rerun()
                    
        # If successfully loaded, show document stats
        if st.session_state.uploaded_doc_text:
            char_count = len(st.session_state.uploaded_doc_text)
            word_count = len(st.session_state.uploaded_doc_text.split())
            st.info(f"📄 **File:** {st.session_state.uploaded_doc_name}\n\n📏 **Size:** {char_count:,} chars (~{word_count:,} words)")
            
            if st.button("🗑️ Clear Document"):
                st.session_state.uploaded_doc_text = None
                st.session_state.uploaded_doc_name = None
                st.session_state.assistant_mode = "Web Search (Tavily)"
                st.rerun()
    else:
        # If file was removed from file uploader, reset session state
        if st.session_state.uploaded_doc_name is not None:
            st.session_state.uploaded_doc_text = None
            st.session_state.uploaded_doc_name = None
            st.session_state.assistant_mode = "Web Search (Tavily)"
            st.rerun()
            
    st.markdown("---")
    st.header("⚙️ Settings")
    
    # Mode selection
    modes = ["Web Search (Tavily)"]
    if st.session_state.uploaded_doc_text:
        modes.append("Ask Uploaded Document")
        
    # Ensure selected mode is valid
    if st.session_state.assistant_mode not in modes:
        st.session_state.assistant_mode = modes[0]
        
    mode = st.radio("Assistant Mode", modes, key="assistant_mode")

c1,c2,c3,c4=st.columns(4)
with c1:
    if st.button("Google"): webbrowser.open_new_tab("https://google.com")
with c2:
    if st.button("YouTube"): webbrowser.open_new_tab("https://youtube.com")
with c3:
    if st.button("LinkedIn"): webbrowser.open_new_tab("https://linkedin.com")
with c4:
    if st.button("GitHub"): webbrowser.open_new_tab("https://github.com")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt=st.chat_input("Ask me anything...")

if prompt:
    st.session_state.show_details=False
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    m=re.search(r"(?:i am|my name is)\s+([a-zA-Z]+)",prompt,re.I)

    if m:
        ans=f"👋 Nice to meet you, {m.group(1).title()}!"
        with st.chat_message("assistant"):
            st.markdown(ans)
        st.session_state.messages.append({"role":"assistant","content":ans})
    else:
        with st.chat_message("assistant"):
            try:
                if st.session_state.assistant_mode == "Ask Uploaded Document":
                    # Retrieve relevant chunks from the vector store
                    chunks = get_relevant_chunks(st.session_state.uploaded_doc_text, prompt)
                    context = "\n---\n".join(chunks)
                    sources = ["Uploaded Document: " + st.session_state.uploaded_doc_name]
                    
                    short_prompt = f"""
Answer the question briefly using the provided document context. If the answer cannot be found in the context, say so.

Document Context:
{context}

Question: {prompt}
"""
                else:
                    results = search.invoke(prompt)
                    context = "\n".join([str(r.get("content","")) for r in results])
                    sources = [r.get("url","") for r in results]

                    short_prompt = f"""
Answer the question briefly using this information.

{context}

Question: {prompt}
"""

                ph=st.empty()
                final=""
                for chunk in llm.stream(short_prompt):
                    if hasattr(chunk,"content"):
                        final+=chunk.content
                        ph.markdown(final+"▌")
                ph.markdown(final)

                detail_prompt = f"""
Explain in detail using the provided context.

Context:
{context}

Question:
{prompt}
"""
                detail=llm.invoke(detail_prompt).content

                st.session_state.report={
                    "detail":detail,
                    "sources":sources
                }

                st.session_state.messages.append(
                    {"role":"assistant","content":final}
                )

            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "rate_limit_exceeded" in err_msg or "Rate limit" in err_msg:
                    st.error("⚠️ AI services are temporarily unavailable. Please try again later.")
                else:
                    st.error(err_msg)

if st.session_state.report:
    if st.button("📖 Show Details"):
        st.session_state.show_details=True

    if st.session_state.show_details:
        st.subheader("Detailed Explanation")
        st.write(st.session_state.report["detail"])

        st.subheader("Sources")
        for s in st.session_state.report["sources"]:
            st.write("-",s)

        st.markdown("---")
        if st.button("📧 Email Report"):
            try:
                sender=os.getenv("EMAIL_USER")
                pwd=os.getenv("EMAIL_PASS")

                msg=EmailMessage()
                msg["Subject"]="AI Report"
                msg["From"]=sender
                msg["To"]=sender
                msg.set_content(st.session_state.report["detail"])

                with smtplib.SMTP_SSL("smtp.gmail.com",465) as smtp:
                    smtp.login(sender,pwd)
                    smtp.send_message(msg)

                st.success("Email sent successfully!")
            except Exception as e:
                st.error(str(e))