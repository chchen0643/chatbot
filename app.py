"""
LangChain + Gemini API 多模態聊天機器人 (Streamlit 版)
支援：圖片、PDF、文字檔、多 Session 管理、即時儲存
"""

import os
import json
import base64
import uuid
import glob
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 載入環境變數
load_dotenv()

# --- 常數設定 ---
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}
TEXT_EXTENSIONS = {"txt", "csv", "md", "json", "xml", "html", "py", "js", "ts", "java", "c", "cpp", "log"}
CHAT_LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_logs")


# ============================
# Session 持久化
# ============================
def ensure_log_dir():
    os.makedirs(CHAT_LOGS_DIR, exist_ok=True)


def get_session_filepath(session_id: str) -> str:
    return os.path.join(CHAT_LOGS_DIR, f"{session_id}.json")


def save_session(session_id: str, messages: list, title: str = ""):
    """即時儲存目前 session 到 JSON"""
    ensure_log_dir()
    data = {
        "session_id": session_id,
        "title": title,
        "updated_at": datetime.now().isoformat(),
        "messages": messages,
    }
    with open(get_session_filepath(session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_session(session_id: str) -> dict | None:
    """載入指定 session"""
    filepath = get_session_filepath(session_id)
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def list_all_sessions() -> list[dict]:
    """列出所有已儲存的 session，按更新時間倒序"""
    ensure_log_dir()
    sessions = []
    for fp in glob.glob(os.path.join(CHAT_LOGS_DIR, "*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "session_id": data.get("session_id", ""),
                "title": data.get("title", "未命名對話"),
                "updated_at": data.get("updated_at", ""),
                "msg_count": len(data.get("messages", [])),
            })
        except Exception:
            continue
    sessions.sort(key=lambda x: x["updated_at"], reverse=True)
    return sessions


def delete_session(session_id: str):
    filepath = get_session_filepath(session_id)
    if os.path.exists(filepath):
        os.remove(filepath)


def generate_title(messages: list) -> str:
    """從第一則使用者訊息產生對話標題"""
    for msg in messages:
        if msg["role"] == "user":
            text = msg["content"]
            # 截取前 30 字作為標題
            return text[:30] + ("..." if len(text) > 30 else "")
    return "新對話"


# ============================
# 模型 & 檔案處理
# ============================
@st.cache_resource
def create_chatbot():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7,
    )


def process_uploaded_file(uploaded_file):
    """將上傳的檔案轉換為 LangChain 可用格式"""
    filename = uploaded_file.name
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    file_bytes = uploaded_file.read()

    if ext in IMAGE_EXTENSIONS:
        mime_type = uploaded_file.type or "image/png"
        data = base64.standard_b64encode(file_bytes).decode("utf-8")
        content_part = {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{data}"},
        }
        return content_part, ("image", file_bytes, filename)

    elif ext == "pdf":
        data = base64.standard_b64encode(file_bytes).decode("utf-8")
        content_part = {
            "type": "media",
            "mime_type": "application/pdf",
            "data": data,
        }
        return content_part, ("pdf", filename)

    else:
        try:
            text_content = file_bytes.decode("utf-8", errors="replace")
            content_part = {
                "type": "text",
                "text": f"--- 檔案: {filename} ---\n{text_content}\n--- 檔案結束 ---",
            }
            return content_part, ("text", filename, text_content)
        except Exception:
            return None, None


# ============================
# 頁面設定
# ============================
st.set_page_config(
    page_title="Gemini 聊天機器人",
    page_icon="🤖",
    layout="centered",
)

st.markdown("""
<style>
    .stMainBlockContainer { max-width: 800px; }
    .block-container { padding-top: 2rem; }
    div[data-testid="stFileUploader"] { margin-bottom: 0; }
    /* 對話列表按鈕樣式 */
    .session-btn { font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)

# --- 初始化 Session State ---
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_title" not in st.session_state:
    st.session_state.session_title = "新對話"


# ============================
# 側邊欄
# ============================
with st.sidebar:
    # --- 新對話按鈕 ---
    if st.button("➕ 開啟新對話", use_container_width=True, type="primary"):
        # 儲存目前對話（如果有內容）
        if st.session_state.messages:
            save_session(
                st.session_state.current_session_id,
                st.session_state.messages,
                st.session_state.session_title,
            )
        # 建立新 session
        st.session_state.current_session_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.session_state.session_title = "新對話"
        st.rerun()

    st.divider()

    # --- System Prompt 選擇 ---
    st.header("🧠 System Prompt")
    from prompt_store import get_all_prompts, add_prompt, delete_prompt

    all_prompts = get_all_prompts()

    if all_prompts:
        prompt_names = [p["name"] for p in all_prompts]
        prompt_map = {p["name"]: p for p in all_prompts}

        selected_prompt_name = st.selectbox(
            "選擇 Prompt",
            options=prompt_names,
            index=0,
            key="selected_prompt",
        )
        active_system_prompt = prompt_map[selected_prompt_name]["prompt"]

        with st.expander("檢視 Prompt 內容"):
            st.markdown(f"```\n{active_system_prompt}\n```")

        # 刪除按鈕
        if st.button(f"🗑️ 刪除「{selected_prompt_name}」", use_container_width=True):
            delete_prompt(prompt_map[selected_prompt_name]["id"])
            st.rerun()
    else:
        active_system_prompt = "你是一個友善的 AI 助手，請用繁體中文回答問題。"
        st.caption("尚無自訂 Prompt，使用預設值")

    # 新增 Prompt
    with st.expander("➕ 新增 System Prompt"):
        new_name = st.text_input("名稱", placeholder="例如：論文助手", key="new_prompt_name")
        new_prompt = st.text_area("Prompt 內容", placeholder="你是一個...", height=100, key="new_prompt_text")
        if st.button("新增", use_container_width=True, key="add_prompt_btn"):
            if new_name.strip() and new_prompt.strip():
                add_prompt(new_name.strip(), new_prompt.strip())
                st.success(f"已新增：{new_name}")
                st.rerun()
            else:
                st.warning("名稱和內容都不能為空")

    st.divider()

    # --- 歷史對話列表 ---
    st.header("💬 對話紀錄")
    all_sessions = list_all_sessions()

    if not all_sessions:
        st.caption("尚無對話紀錄")
    else:
        for sess in all_sessions:
            sid = sess["session_id"]
            is_current = sid == st.session_state.current_session_id
            title = sess["title"] or "未命名對話"
            msg_count = sess["msg_count"]

            # 顯示時間
            try:
                dt = datetime.fromisoformat(sess["updated_at"])
                time_str = dt.strftime("%m/%d %H:%M")
            except Exception:
                time_str = ""

            col1, col2 = st.columns([5, 1])
            with col1:
                label = f"{'▶ ' if is_current else ''}{title}"
                if st.button(
                    label,
                    key=f"sess_{sid}",
                    use_container_width=True,
                    disabled=is_current,
                ):
                    # 儲存目前對話再切換
                    if st.session_state.messages:
                        save_session(
                            st.session_state.current_session_id,
                            st.session_state.messages,
                            st.session_state.session_title,
                        )
                    # 載入選取的 session
                    loaded = load_session(sid)
                    if loaded:
                        st.session_state.current_session_id = sid
                        st.session_state.messages = loaded.get("messages", [])
                        st.session_state.session_title = loaded.get("title", "未命名對話")
                    st.rerun()

            with col2:
                if st.button("🗑", key=f"del_{sid}"):
                    delete_session(sid)
                    if is_current:
                        st.session_state.current_session_id = str(uuid.uuid4())[:8]
                        st.session_state.messages = []
                        st.session_state.session_title = "新對話"
                    st.rerun()

            # 副標題：時間 + 訊息數
            if not is_current:
                st.caption(f"    {time_str} · {msg_count} 則訊息")

    st.divider()

    # --- 檔案上傳 ---
    st.header("📎 上傳檔案")
    uploaded_files = st.file_uploader(
        "拖放或選擇檔案",
        accept_multiple_files=True,
        type=list(IMAGE_EXTENSIONS | TEXT_EXTENSIONS | {"pdf"}),
        label_visibility="collapsed",
    )

    selected_file_names = []
    if uploaded_files:
        file_options = []
        for uf in uploaded_files:
            ext = uf.name.rsplit(".", 1)[-1].lower() if "." in uf.name else ""
            if ext in IMAGE_EXTENSIONS:
                icon = "📷"
            elif ext == "pdf":
                icon = "📄"
            else:
                icon = "📝"
            file_options.append(f"{icon} {uf.name}")

        selected_file_names = st.multiselect(
            "選擇要附上的檔案",
            options=file_options,
            default=[],
            placeholder="選擇檔案...",
        )


# ============================
# 主畫面
# ============================
st.title("🤖 Gemini 多模態聊天機器人")
st.caption("支援文字對話、圖片分析、PDF 摘要、文字檔解讀")

# --- 顯示歷史訊息 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "images" in msg:
            for img_info in msg["images"]:
                # images 存的是 (base64_data, filename) 方便 JSON 序列化
                st.image(base64.b64decode(img_info["data"]), caption=img_info["name"], use_container_width=True)

# --- 使用者輸入 ---
user_text = st.chat_input("輸入訊息...")

if user_text:
    llm = create_chatbot()

    # 處理選取的檔案
    content_parts = []
    display_images = []  # 用於即時顯示
    serializable_images = []  # 用於 JSON 儲存
    file_names = []

    if uploaded_files and selected_file_names:
        for uf in uploaded_files:
            ext = uf.name.rsplit(".", 1)[-1].lower() if "." in uf.name else ""
            if ext in IMAGE_EXTENSIONS:
                icon = "📷"
            elif ext == "pdf":
                icon = "📄"
            else:
                icon = "📝"
            label = f"{icon} {uf.name}"
            if label not in selected_file_names:
                continue
            uf.seek(0)
            part, info = process_uploaded_file(uf)
            if part:
                content_parts.append(part)
                if info[0] == "image":
                    img_b64 = base64.standard_b64encode(info[1]).decode("utf-8")
                    display_images.append((info[1], info[2]))
                    serializable_images.append({"data": img_b64, "name": info[2]})
                    file_names.append(f"📷 {info[2]}")
                elif info[0] == "pdf":
                    file_names.append(f"📄 {info[1]}")
                elif info[0] == "text":
                    file_names.append(f"📝 {info[1]}")

    # 組合顯示文字
    display_text = user_text
    if file_names:
        display_text += "\n\n**附件：** " + "、".join(file_names)

    # 顯示使用者訊息
    with st.chat_message("user"):
        st.markdown(display_text)
        for img_bytes, img_name in display_images:
            st.image(img_bytes, caption=img_name, use_container_width=True)

    # 儲存使用者訊息
    user_msg_record = {"role": "user", "content": display_text}
    if serializable_images:
        user_msg_record["images"] = serializable_images
    st.session_state.messages.append(user_msg_record)

    # 更新標題（用第一則訊息）
    if st.session_state.session_title == "新對話":
        st.session_state.session_title = generate_title(st.session_state.messages)

    # 建立 LangChain 訊息
    content_parts.append({"type": "text", "text": user_text})
    human_msg = HumanMessage(content=content_parts)

    lc_messages = [SystemMessage(content=active_system_prompt)]
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))
    lc_messages.append(human_msg)

    # 呼叫模型
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                response = llm.invoke(lc_messages)
                ai_reply = response.content
                st.markdown(ai_reply)
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            except Exception as e:
                error_msg = f"發生錯誤: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": f"❌ {error_msg}"})

    # === 即時儲存 ===
    save_session(
        st.session_state.current_session_id,
        st.session_state.messages,
        st.session_state.session_title,
    )
