"""
System Prompt 持久化管理模組
使用 JSON 檔案儲存所有自訂的 system prompts
"""

import os
import json
import uuid

PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_prompts.json")

DEFAULT_PROMPTS = [
    {
        "id": "default",
        "name": "預設助手",
        "prompt": "你是一個友善的 AI 助手，請用繁體中文回答問題。你可以分析圖片、PDF 和文字檔案。",
    },
    {
        "id": "translator",
        "name": "翻譯助手",
        "prompt": "你是一個專業的翻譯助手。請將使用者的輸入翻譯成目標語言，並保持原文的語意和語氣。如果未指定目標語言，請翻譯成英文。",
    },
    {
        "id": "code_reviewer",
        "name": "程式碼審查",
        "prompt": "你是一位資深軟體工程師，專門進行程式碼審查。請分析使用者提供的程式碼，指出潛在問題、安全漏洞、效能瓶頸，並提供改善建議。請用繁體中文回答。",
    },
]


def _load_prompts_file() -> list[dict]:
    """從 JSON 檔案載入 prompts"""
    if not os.path.exists(PROMPTS_FILE):
        # 初次使用，寫入預設值
        _save_prompts_file(DEFAULT_PROMPTS)
        return DEFAULT_PROMPTS

    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_prompts_file(prompts: list[dict]):
    """儲存 prompts 到 JSON 檔案"""
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)


def get_all_prompts() -> list[dict]:
    """取得所有 system prompts"""
    return _load_prompts_file()


def get_prompt_by_id(prompt_id: str) -> dict | None:
    """依 ID 取得 prompt"""
    for p in _load_prompts_file():
        if p["id"] == prompt_id:
            return p
    return None


def add_prompt(name: str, prompt: str) -> dict:
    """新增一個 system prompt"""
    prompts = _load_prompts_file()
    new_prompt = {
        "id": str(uuid.uuid4())[:8],
        "name": name,
        "prompt": prompt,
    }
    prompts.append(new_prompt)
    _save_prompts_file(prompts)
    return new_prompt


def update_prompt(prompt_id: str, name: str, prompt: str) -> bool:
    """更新指定 prompt"""
    prompts = _load_prompts_file()
    for p in prompts:
        if p["id"] == prompt_id:
            p["name"] = name
            p["prompt"] = prompt
            _save_prompts_file(prompts)
            return True
    return False


def delete_prompt(prompt_id: str) -> bool:
    """刪除指定 prompt"""
    prompts = _load_prompts_file()
    new_prompts = [p for p in prompts if p["id"] != prompt_id]
    if len(new_prompts) < len(prompts):
        _save_prompts_file(new_prompts)
        return True
    return False
