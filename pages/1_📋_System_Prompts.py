"""
System Prompt 管理頁面
新增、編輯、刪除自訂的 System Prompts
"""

import streamlit as st
import sys
import os

# 確保可以 import 根目錄模組
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from prompt_store import get_all_prompts, add_prompt, update_prompt, delete_prompt

st.set_page_config(page_title="System Prompt 管理", page_icon="📋", layout="centered")

st.title("📋 System Prompt 管理")
st.caption("新增、編輯、刪除你的 System Prompts，在聊天時可以自由切換")

# --- 新增 Prompt ---
st.header("➕ 新增 Prompt")

with st.form("add_prompt_form", clear_on_submit=True):
    new_name = st.text_input("名稱", placeholder="例如：論文助手")
    new_prompt = st.text_area("Prompt 內容", placeholder="你是一個...", height=120)
    submitted = st.form_submit_button("新增", use_container_width=True, type="primary")

    if submitted:
        if new_name.strip() and new_prompt.strip():
            add_prompt(new_name.strip(), new_prompt.strip())
            st.success(f"已新增：{new_name}")
            st.rerun()
        else:
            st.warning("名稱和內容都不能為空")

st.divider()

# --- 已有的 Prompts ---
st.header("📝 現有 Prompts")

prompts = get_all_prompts()

if not prompts:
    st.info("尚無任何 System Prompt，請先新增一個。")
else:
    for i, p in enumerate(prompts):
        with st.expander(f"**{p['name']}**", expanded=False):
            # 編輯表單
            with st.form(f"edit_{p['id']}"):
                edit_name = st.text_input("名稱", value=p["name"], key=f"name_{p['id']}")
                edit_prompt = st.text_area("Prompt 內容", value=p["prompt"], height=150, key=f"prompt_{p['id']}")

                col1, col2 = st.columns(2)
                with col1:
                    save_btn = st.form_submit_button("💾 儲存修改", use_container_width=True)
                with col2:
                    del_btn = st.form_submit_button("🗑️ 刪除", use_container_width=True)

                if save_btn:
                    if edit_name.strip() and edit_prompt.strip():
                        update_prompt(p["id"], edit_name.strip(), edit_prompt.strip())
                        st.success("已儲存修改！")
                        st.rerun()
                    else:
                        st.warning("名稱和內容都不能為空")

                if del_btn:
                    delete_prompt(p["id"])
                    st.success(f"已刪除：{p['name']}")
                    st.rerun()
