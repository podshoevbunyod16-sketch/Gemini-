import streamlit as st
import google.generativeai as genai

# --- НАСТРОЙКИ ---
# Вставьте свой ключ ВНУТРЬ кавычек ниже!
API_KEY = "AIzaSyBjuTWH3hRJhUI1ViloRdfMx4q6WChHPbQ" 

st.set_page_config(page_title="Gemini Mobile", page_icon="🤖")

# Заголовок
st.title("🤖 Gemini в Termux")

# История сообщений
if "messages" not in st.session_state:
    st.session_state.messages = []

# Показываем переписку
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Поле ввода
prompt = st.chat_input("Напишите сообщение...")

if prompt:
    # 1. Показываем вопрос пользователя
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Получаем ответ от Gemini
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                response = model.generate_content(prompt)
                st.write(response.text)
        
        # Сохраняем ответ
        st.session_state.messages.append({"role": "assistant", "content": response.text})
        
    except Exception as e:
        st.error(f"Ошибка: {e}")
