"""
LangChain + Gemini API 多模態聊天機器人
支援：圖片（jpg/png/gif/webp）、PDF、文字檔（txt/csv/md 等）
"""

import os
import re
import base64
import mimetypes
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 載入環境變數
load_dotenv()

# 支援的檔案類型
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
TEXT_EXTENSIONS = {".txt", ".csv", ".md", ".json", ".xml", ".html", ".py", ".js", ".ts", ".java", ".c", ".cpp", ".log"}
PDF_EXTENSION = ".pdf"


def parse_input(user_input: str):
    """
    解析使用者輸入，分離文字訊息與檔案路徑。
    使用者可用 @filepath 的方式附加檔案，例如：
      請分析這張圖 @/path/to/image.png
      @/path/to/doc.pdf 幫我摘要這份文件
    """
    # 匹配 @ 後面的檔案路徑（支援空白用引號包住）
    file_pattern = r'@"([^"]+)"|@(\S+)'
    files = []
    for match in re.finditer(file_pattern, user_input):
        filepath = match.group(1) or match.group(2)
        filepath = os.path.expanduser(filepath)
        if os.path.isfile(filepath):
            files.append(filepath)
        else:
            print(f"  ⚠️  找不到檔案: {filepath}")

    # 移除 @filepath 部分，保留純文字
    text = re.sub(file_pattern, "", user_input).strip()
    return text, files


def load_file_content(filepath: str):
    """將檔案轉換為 LangChain 可用的訊息內容格式"""
    ext = os.path.splitext(filepath)[1].lower()
    filename = os.path.basename(filepath)

    if ext in IMAGE_EXTENSIONS:
        mime_type = mimetypes.guess_type(filepath)[0] or "image/png"
        with open(filepath, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        print(f"  📷 已載入圖片: {filename}")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{data}"},
        }

    elif ext == PDF_EXTENSION:
        mime_type = "application/pdf"
        with open(filepath, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        print(f"  📄 已載入 PDF: {filename}")
        return {
            "type": "media",
            "mime_type": mime_type,
            "data": data,
        }

    elif ext in TEXT_EXTENSIONS:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        print(f"  📝 已載入文字檔: {filename}")
        return {
            "type": "text",
            "text": f"--- 檔案: {filename} ---\n{content}\n--- 檔案結束 ---",
        }

    else:
        # 嘗試以文字方式讀取
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            print(f"  📎 已載入檔案: {filename}")
            return {
                "type": "text",
                "text": f"--- 檔案: {filename} ---\n{content}\n--- 檔案結束 ---",
            }
        except Exception:
            print(f"  ❌ 無法讀取檔案: {filename}")
            return None


def create_chatbot():
    """建立 Gemini 聊天機器人"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7,
    )
    return llm


def build_human_message(text: str, files: list):
    """建立包含文字和檔案的 HumanMessage"""
    content_parts = []

    # 加入檔案內容
    for filepath in files:
        part = load_file_content(filepath)
        if part:
            content_parts.append(part)

    # 加入文字訊息
    if text:
        content_parts.append({"type": "text", "text": text})
    elif not content_parts:
        return None

    # 如果沒有文字但有檔案，加個預設提示
    if not text and content_parts:
        content_parts.append({"type": "text", "text": "請分析上述內容。"})

    return HumanMessage(content=content_parts)


def save_chat_history(chat_history: list):
    """將對話紀錄儲存為 Markdown 檔案"""
    if not chat_history:
        print("\n（本次沒有對話紀錄，不儲存）")
        return

    # 建立 chat_logs 資料夾
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_logs")
    os.makedirs(log_dir, exist_ok=True)

    # 用時間戳命名檔案
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_{timestamp}.md"
    filepath = os.path.join(log_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# 對話紀錄\n\n")
        f.write(f"**日期時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                content = msg.content if isinstance(msg.content, str) else "(多模態訊息)"
                f.write(f"## 🧑 你\n\n{content}\n\n")
            elif isinstance(msg, AIMessage):
                f.write(f"## 🤖 AI\n\n{msg.content}\n\n")
            f.write("---\n\n")

    print(f"\n💾 對話紀錄已儲存至: {filepath}")


def main():
    llm = create_chatbot()
    chat_history = []

    system_msg = SystemMessage(content="你是一個友善的 AI 助手，請用繁體中文回答問題。你可以分析圖片、PDF 和文字檔案。")

    print("=" * 56)
    print("  Gemini 多模態聊天機器人")
    print("  • 輸入文字直接對話")
    print("  • 用 @檔案路徑 附加檔案（圖片/PDF/文字檔）")
    print('  • 路徑有空白請用引號：@"/path/my file.pdf"')
    print("  • 輸入 'quit' 或 'exit' 離開")
    print("=" * 56)

    try:
      while True:
        user_input = input("\n你: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        # 解析輸入
        text, files = parse_input(user_input)

        # 建立訊息
        human_msg = build_human_message(text, files)
        if human_msg is None:
            continue

        # 組合完整訊息（system + 歷史 + 當前）
        messages = [system_msg] + chat_history + [human_msg]

        try:
            response = llm.invoke(messages)
            ai_reply = response.content
            print(f"\nAI: {ai_reply}")

            # 更新對話歷史（純文字版本以節省 token）
            history_text = text if text else "(使用者傳送了檔案)"
            if files:
                filenames = [os.path.basename(f) for f in files]
                history_text += f" [附件: {', '.join(filenames)}]"
            chat_history.append(HumanMessage(content=history_text))
            chat_history.append(AIMessage(content=ai_reply))

        except Exception as e:
            print(f"\n❌ 錯誤: {e}")

    except KeyboardInterrupt:
        pass  # Ctrl+C 也會儲存紀錄

    # 結束時儲存對話紀錄
    save_chat_history(chat_history)
    print("\n再見！👋")


if __name__ == "__main__":
    main()
