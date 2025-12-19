import streamlit as st
import uuid
import pandas as pd
import json
import requests
import jwt
import urllib.parse
from api_client import APIClient

import os

st.set_page_config(page_title="DemoBot", layout="wide")

# Configuration from Environment Variables or Secrets
COGNITO_DOMAIN = os.getenv("COGNITO_DOMAIN", "https://your-cognito-domain.auth.us-east-1.amazoncognito.com")
CLIENT_ID = os.getenv("COGNITO_CLIENT_ID", "your-client-id")
REDIRECT_URI = os.getenv("App_REDIRECT_URI", "http://localhost:8501")

# -------------------------------------------------
# AUTH FUNCTIONS
# -------------------------------------------------
def exchange_code_for_token(code):
    token_url = f"{COGNITO_DOMAIN}/oauth2/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": code,
        "redirect_uri": REDIRECT_URI,
    }
    try:
        response = requests.post(token_url, headers=headers, data=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:

        st.error(f"Token exchange failed: {e}")
        return None

def decode_id_token(id_token):
    try:
        
        decoded = jwt.decode(id_token, options={"verify_signature": False})
        return decoded.get("email", "Unknown")
    except Exception as e:
        st.error(f"Token decoding failed: {e}")
        return "Unknown"

# -------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "id_token" not in st.session_state:
    st.session_state.id_token = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None # Initially None until auth

# Initialize persistence for Chat Sessions
if "all_chats" not in st.session_state:
    st.session_state.all_chats = []

# Initialize Current Chat
if "chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.chat_id = new_id
    st.session_state.messages = []
    st.session_state.all_chats.insert(0, {"id": new_id, "title": "New Chat", "messages": []})

if "api" not in st.session_state:
    # We initialize with a placeholder, will update user_id after auth
    st.session_state.api = APIClient("guest")
else:
    # Hot-fix for development: Re-init to load new methods if missing
    if not hasattr(st.session_state.api, "delete_chat"):
         st.session_state.api = APIClient("guest")
         if st.session_state.user_id:
             st.session_state.api.set_user(st.session_state.user_id)

# ===== Helper Functions =====
def refresh_chat_list():
    if st.session_state.get("user_id"):
        chats = st.session_state.api.list_chats()
        if chats:
            # Map backend format to frontend format
            st.session_state.all_chats = []
            for c in chats:
                st.session_state.all_chats.append({
                    "id": c["id"],
                    "title": c.get("title", f"Chat {c.get('timestamp', '')[:10]}"),
                    "messages": [] # Loaded on demand
                })
        else:
             if not st.session_state.all_chats:
                 # Default new chat if absolutely nothing
                 create_new_chat(rerun=False)

def create_new_chat(rerun=True):
    new_id = str(uuid.uuid4())
    st.session_state.chat_id = new_id
    st.session_state.messages = []
    st.session_state.all_chats.insert(0, {"id": new_id, "title": "New Chat", "messages": []})
    if rerun: st.rerun()

def load_chat(chat_data):
    st.session_state.chat_id = chat_data["id"]
    # Fetch history from backend
    history = st.session_state.api.get_chat_history(chat_data["id"])
    if history:
        st.session_state.messages = history
        chat_data["messages"] = history
    else:
        st.session_state.messages = []
    st.rerun()

def delete_chat(chat_id):
    # Call Backend Delete
    with st.spinner("Deleting..."):
        resp = st.session_state.api.delete_chat(chat_id)
        if resp.get("success"):
            st.session_state.all_chats = [c for c in st.session_state.all_chats if c['id'] != chat_id]
            st.toast(f"Chat deleted")
            if st.session_state.chat_id == chat_id:
                create_new_chat()
            else:
                st.rerun()
        else:
            st.error(f"Failed to delete: {resp.get('errorMessage')}")

def save_current_chat():
    for chat in st.session_state.all_chats:
        if chat["id"] == st.session_state.chat_id:
            chat["messages"] = st.session_state.messages
            if chat["title"] == "New Chat" and len(st.session_state.messages) > 0:
                first_msg = next((m for m in st.session_state.messages if m["role"] == "user"), None)
                if first_msg:
                    chat["title"] = first_msg["content"][:30] + "..."
            break

# -------------------------------------------------
# HANDLE AUTH REDIRECT
# -------------------------------------------------
query_params = st.query_params
if "code" in query_params and not st.session_state.access_token:
    tokens = exchange_code_for_token(query_params["code"])
    if tokens:
        st.session_state.access_token = tokens.get("access_token")
        st.session_state.id_token = tokens.get("id_token")
        
        if st.session_state.id_token:
            email = decode_id_token(st.session_state.id_token)
            st.session_state.user_id = email
            st.session_state.api.set_user(email) # Update API client
            st.session_state.api.set_token(st.session_state.id_token) # Set Auth Token
            refresh_chat_list() # Load previous chats
            
        st.query_params.clear()
        st.rerun()

# -------------------------------------------------
# MAIN APP LOGIC
# -------------------------------------------------

# CSS
st.markdown("""
<style>
    .stChatInput { bottom: 20px; }
    .block-container { padding-top: 2rem; }
    section[data-testid="stSidebar"] { background-color: #0e1117; color: #fafafa; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] div, section[data-testid="stSidebar"] label { color: #fafafa !important; }
    section[data-testid="stSidebar"] .stButton button { background-color: #262730; color: white; border: 1px solid #4b4b4b; }
    section[data-testid="stSidebar"] .stButton button:hover { background-color: #ff4b4b; border-color: #ff4b4b; color: white; }
    .stButton button { width: 100%; }
    [data-testid="stChatMessageAvatar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# CHECK AUTH STATE
if not st.session_state.access_token:
    # Not Authenticated - Show Login
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("Sign In Required")
        st.markdown("Please sign in with your corporate identity to access the DemoBot System.")
        
        signin_url = (
            f"{COGNITO_DOMAIN}/login"
            f"?client_id={CLIENT_ID}"
            f"&response_type=code"
            f"&scope=email+openid+phone"
            f"&redirect_uri={urllib.parse.quote(REDIRECT_URI)}"
        )
        
        st.markdown(
            f"<a href='{signin_url}' target='_self'><button style='padding: 0.5rem 1rem; background-color: #FF4B4B; color: white; border: none; border-radius: 4px; cursor: pointer; width: 100%; font-size: 1.2rem;'>Sign In with Cognito</button></a>",
            unsafe_allow_html=True
        )
else:
    # Authenticated - Show App
    
    # Ensure API client has correct user & token (double check)
    if st.session_state.api.user_id != st.session_state.user_id:
        st.session_state.api.set_user(st.session_state.user_id)
    
    # Always ensure token is set if we are authenticated
    if st.session_state.id_token:
        st.session_state.api.set_token(st.session_state.id_token)

    # ===== Sidebar Functionality =====
    with st.sidebar:
        st.title("DemoBot")
        st.caption(f"User: `{st.session_state.user_id}`")
        if st.button("Sign Out"):
            st.session_state.access_token = None
            st.session_state.id_token = None
            st.session_state.user_id = None
            st.rerun()
        
        st.divider()

        # 2. Chat Management
        if st.button("New Chat", type="primary"):
            create_new_chat()
        
        st.subheader("Recent Chats")
        
        # List chats
        for chat in st.session_state.all_chats:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                # Highlight current chat
                is_active = (chat["id"] == st.session_state.chat_id)
                label = f"{'> ' if is_active else ''}{chat['title']}"
                if st.button(label, key=f"btn_{chat['id']}"):
                    load_chat(chat)
            with col2:
                if st.button("Del", key=f"del_{chat['id']}", help="Delete Chat"):
                    delete_chat(chat["id"])
        
        st.divider()

        # 3. Knowledge Base Management
        st.subheader("Knowledge Base")
        
        tab1, tab2 = st.tabs(["Upload", "Manage"])
        
        with tab1:
            uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt", "png", "jpg", "jpeg"])
            if uploaded_file and st.button("Index Document"):
                with st.spinner("Uploading to S3 & Indexing..."):
                    success, msg = st.session_state.api.upload_file(
                        uploaded_file.getvalue(), 
                        uploaded_file.name,
                        chat_id=st.session_state.chat_id, # Scope to current chat
                        doc_title=uploaded_file.name.split('.')[0]
                    )
                    if success:
                        st.success("Indexed!")
                        # Check if it's an image to display
                        file_type = uploaded_file.name.split('.')[-1].lower()
                        if file_type in ['png', 'jpg', 'jpeg']:
                             st.session_state.messages.append({
                                 "role": "user", 
                                 "content": f"I have uploaded an image: `{uploaded_file.name}`. [Image Uploaded]",
                                 "image": uploaded_file.getvalue()
                             })
                        else:
                             st.session_state.messages.append({
                                 "role": "user", 
                                 "content": f"I have uploaded a document: `{uploaded_file.name}`."
                             })
                        save_current_chat()
                        st.rerun()
                    else:
                        st.error(f"Error: {msg}")

        with tab2:
            if st.button("Refresh Files"):
                 pass 
                 
            files = st.session_state.api.list_files()
            if files:
                # Create cleaner dataframe
                data = []
                for f in files:
                    data.append({
                        "Name": f.get('doc_id', 'Unknown'),
                        "Time": f.get('upload_timestamp', '')[:10], # Just date
                        "Key": f.get('index_key')
                    })
                
                df = pd.DataFrame(data)
                if not df.empty:
                    st.dataframe(df[['Name', 'Time']], hide_index=True, use_container_width=True)
                    
                    # Delete interaction
                    to_delete = st.selectbox("Select file to delete:", df['Key'].tolist(), format_func=lambda x: next((f['Name'] for f in data if f['Key'] == x), x), key="del_select")
                    if st.button(f"Delete Selected"):
                        with st.spinner("Deleting..."):
                            del_res = st.session_state.api.delete_file(to_delete)
                            if del_res.get("success"):
                                st.success("Deleted!")
                                st.rerun()
                            else:
                                st.error(str(del_res))
            else:
                st.info("No files found.")
        
        st.divider()
        
        # Feature Toggles - Optional Enhancements
        st.subheader("Additional Sources")
        st.caption("General conversation is always available. Enable sources below to enhance responses:")
        
        # Smart Mode - Backend decides
        use_smart_mode = st.checkbox(
            "Smart Mode", 
            value=True, 
            help="Let the underlying AI model decide which tools to use based on your query",
            key="feat_smart_mode"
        )

        st.divider()
        st.caption("Detailed Controls (Manual / Permissions):")
        
        # Document Agent - Intelligent chat document retrieval
        use_doc_agent = st.checkbox(
            "Document Agent", 
            value=True, 
            help="Enable access to your uploaded documents",
            key="feat_doc_agent"
        )
        
        # KB - Global admin policies
        use_kb = st.checkbox(
            "Knowledge Base", 
            value=False, 
            help="Enable access to company-wide policies (Admin)",
            key="feat_kb"
        )
        
        # Web Search
        use_websearch = st.checkbox(
            "Web Search", 
            value=False, 
            help="Enable internet search (Only if checked)",
            key="feat_web"
        )
        
        # Show active sources
        sources = ["Chat"]  # Always available
        if use_smart_mode:
            sources.append("Smart Mode")
        else:
             if use_doc_agent: sources.append("Doc Agent")
             if use_kb: sources.append("KB")
             if use_websearch: sources.append("Web")
        
        st.info(f"Active: {' + '.join(sources)}")
        
        # Store in session state
        st.session_state.features = {
            "smart_mode": use_smart_mode,
            "use_doc_agent": use_doc_agent, 
            "use_kb": use_kb,  
            "use_websearch": use_websearch,
            "doc_chat_only": not use_kb 
        }

    # ===== Main Chat Area =====
    # Find current chat title for header
    current_title = next((c["title"] for c in st.session_state.all_chats if c["id"] == st.session_state.chat_id), "New Chat")
    st.title(current_title)
    
    # Display History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "image" in msg:
                st.image(msg["image"], caption="Uploaded Image", use_column_width=True)

    # Chat Input
    if prompt := st.chat_input("Start typing your query..."):
        # 1. User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_current_chat() # Save immediately
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # 2. Assistant Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Processing..."):
                try:
                    response = st.session_state.api.query_rag(
                        st.session_state.chat_id, 
                        prompt,
                        features=st.session_state.get("features", {
                            "smart_mode": True,
                            "use_doc_agent": True,
                            "use_kb": True,
                            "use_websearch": False,
                            "doc_chat_only": True
                        })
                    )
                    
                    # Robust parsing of the Lambda response
                    final_text = ""
                    
                    # 1. unwrapping "result" from API response
                    payload = response
                    if isinstance(response, dict) and response.get("success"):
                        payload = response.get("result", {})
                    
                    # 2. Unwrap "response" field if present (from query_rag_bedrock.py structure)
                    if isinstance(payload, dict) and "response" in payload:
                        payload = payload["response"]
                        
                    # 3. Handle Bedrock/Agentic Object Structure
                    # Structure: {'role': 'assistant', 'content': [{'text': '...'}]}
                    if isinstance(payload, dict):
                        if "content" in payload and isinstance(payload["content"], list):
                            # Extract text from Bedrock content blocks
                            texts = []
                            for block in payload["content"]:
                                if "text" in block:
                                    texts.append(block["text"])
                            final_text = "".join(texts)
                        elif "answer" in payload:
                            final_text = payload["answer"]
                        elif "message" in payload:
                            final_text = payload["message"]
                        elif "text" in payload:
                            final_text = payload["text"]
                        else:
                            # Fallback: dump json if we can't find a text field
                            final_text = str(payload)
                    else:
                        # It's a string (or simple type)
                        final_text = str(payload)

                    if isinstance(final_text, str):
                        # Decode if it's a JSON string by accident
                        try:
                            parsed = json.loads(final_text)
                            if isinstance(parsed, dict) and "content" in parsed:
                                # Recursively handle if double-encoded
                                blocks = parsed.get("content", [])
                                final_text = "".join([b.get("text","") for b in blocks])
                        except:
                            pass

                    message_placeholder.markdown(final_text)
                    st.session_state.messages.append({"role": "assistant", "content": final_text})
                    save_current_chat() # Save response
                    
                except Exception as e:
                    st.error(f"Connection Error: {e}")
