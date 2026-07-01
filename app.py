"""
Веб-приложение для RAG-системы с локальной ИИ-моделью
"""

import streamlit as st
import os
import tempfile
import time
import shutil
from rag_system import RAGSystem, RAGWithLLM

# === НАСТРОЙКА ВРЕМЕННЫХ ФАЙЛОВ ===
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

def cleanup_old_files():
    """Очистка старых временных файлов"""
    try:
        for file in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, file)
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except:
                pass
    except:
        pass

# Очищаем при запуске
cleanup_old_files()

# === НАСТРОЙКА СТРАНИЦЫ ===
st.set_page_config(
    page_title="RAG-поиск с локальной ИИ-моделью",
    page_icon="🤖",
    layout="wide"
)

# === ИНИЦИАЛИЗАЦИЯ ===
if 'rag_system' not in st.session_state:
    with st.spinner("🔄 Загрузка системы..."):
        st.session_state.rag_system = RAGSystem()
        st.session_state.rag_with_llm = RAGWithLLM(st.session_state.rag_system)
        
        # Создаем тестовый документ если его нет
        test_doc = os.path.join(TEMP_DIR, "test_document.txt")
        if not os.path.exists(test_doc):
            with open(test_doc, "w", encoding="utf-8") as f:
                f.write("""
ПОЛОЖЕНИЕ О КОМАНДИРОВКАХ

1. Общие положения
1.1. Настоящее положение определяет порядок направления сотрудников в служебные командировки.
1.2. Командировка оформляется приказом генерального директора.

2. Порядок оформления
2.1. Сотрудник подает заявление не позднее чем за 3 дня до командировки.
2.2. Заявление согласовывается с руководителем подразделения.

3. Оплата расходов
3.1. Суточные выплачиваются в размере 1000 рублей за каждый день командировки.
3.2. Расходы на проезд компенсируются по фактическим затратам.
""")
            st.session_state.rag_system.add_document(test_doc)
            st.success("✅ Создан тестовый документ")
        
        # Удаляем тестовый документ после загрузки
        try:
            if os.path.exists(test_doc):
                os.unlink(test_doc)
        except:
            pass

if 'messages' not in st.session_state:
    st.session_state.messages = []

# === ЗАГОЛОВОК ===
st.title("🤖 Интеллектуальный помощник с локальной ИИ-моделью")
st.markdown("Загружайте документы и задавайте вопросы — система найдёт ответ в вашей базе")

# === БОКОВАЯ ПАНЕЛЬ ===
with st.sidebar:
    st.header("⚙️ Управление документами")
    
    # Загрузка файлов
    uploaded_files = st.file_uploader(
        "Загрузите документы",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("📥 Добавить в базу", type="primary"):
            with st.spinner("Обработка документов..."):
                total_chunks = 0
                failed_files = []
                
                for file in uploaded_files:
                    try:
                        # Сохраняем файл во временную папку
                        safe_filename = f"{int(time.time() * 1000)}_{file.name}"
                        file_path = os.path.join(TEMP_DIR, safe_filename)
                        
                        with open(file_path, 'wb') as f:
                            f.write(file.getvalue())
                        
                        # Добавляем в RAG
                        chunks = st.session_state.rag_system.add_document(file_path)
                        total_chunks += chunks
                        
                        # Удаляем файл с повторными попытками
                        deleted = False
                        for attempt in range(5):
                            try:
                                if os.path.exists(file_path):
                                    os.unlink(file_path)
                                    deleted = True
                                    break
                            except (PermissionError, OSError):
                                time.sleep(0.3)
                        
                        if not deleted:
                            # Если не удалось удалить, попробуем позже
                            st.warning(f"Файл {file.name} обработан, но не удален")
                            
                    except Exception as e:
                        failed_files.append(f"{file.name}: {str(e)}")
                
                # Сообщаем о результатах
                if failed_files:
                    st.warning(f"⚠️ Ошибки при обработке: {', '.join(failed_files)}")
                else:
                    st.success(f"✅ Добавлено {total_chunks} блоков!")
                
                time.sleep(1)
                st.rerun()
    
    st.divider()
    
    # === НАСТРОЙКИ МОДЕЛИ ===
    st.header("🤖 Настройки модели")
    
    generation_mode = st.selectbox(
        "Режим работы:",
        [
            "Простой поиск (без ИИ)",
            "Локальная ИИ-модель (рекомендуется)",
            "OpenAI API (требуется ключ)"
        ]
    )
    
    if generation_mode == "Локальная ИИ-модель (рекомендуется)":
        st.info("💡 Используется бесплатная локальная модель")
        
        model_options = {
            "microsoft/phi-2": "Phi-2 (легкая, быстрая)",
            "google/gemma-2b-it": "Gemma-2B (компактная)",
            "Qwen/Qwen-1_8B-Chat": "Qwen-1.8B (для русского)",
            "HuggingFaceH4/zephyr-7b-beta": "Zephyr-7B (качественная)",
            "mistralai/Mistral-7B-Instruct-v0.1": "Mistral-7B (лучшая)"
        }
        
        selected_model = st.selectbox(
            "Выберите модель:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0
        )
        
        col1, col2 = st.columns(2)
        with col1:
            use_gpu = st.checkbox("Использовать GPU", value=False)
        with col2:
            quantize = st.checkbox("Квантизация", value=True)
        
        if st.button("🔄 Загрузить модель"):
            with st.spinner("Загрузка модели... Это может занять несколько минут"):
                try:
                    result = st.session_state.rag_with_llm.get_answer_local(
                        "Тестовый запрос",
                        model_name=selected_model,
                        use_gpu=use_gpu,
                        quantize=quantize
                    )
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.success("✅ Модель успешно загружена!")
                        if "model_info" in result:
                            st.json(result["model_info"])
                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")
    
    if generation_mode == "OpenAI API (требуется ключ)":
        openai_key = st.text_input("Введите OpenAI API Key:", type="password")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
    
    st.divider()
    
    # === СТАТИСТИКА ===
    st.header("📊 Статистика")
    stats = st.session_state.rag_system.get_database_stats()
    st.metric("Блоков в базе", stats["total_chunks"])
    st.metric("Размерность векторов", stats["vector_dimension"])
    
    if st.button("🗑️ Очистить базу", type="secondary"):
        st.session_state.rag_system.clear_database()
        st.success("База очищена!")
        st.rerun()

# === ОСНОВНАЯ ОБЛАСТЬ ===
st.header("💬 Задайте вопрос")

# История чата
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "sources" in message and message["sources"]:
            with st.expander("📄 Источники"):
                for i, source in enumerate(message["sources"], 1):
                    st.write(f"**{i}. {source['document']}**")
                    st.write(f"Релевантность: {source['score']:.2%}")
                    st.write(f"_{source['text'][:200]}..._")
                    st.divider()

# === ПОЛЕ ВВОДА ===
if prompt := st.chat_input("Введите ваш вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Поиск информации..."):
            
            if generation_mode == "Простой поиск (без ИИ)":
                result = st.session_state.rag_with_llm.get_answer_simple(prompt)
            
            elif generation_mode == "Локальная ИИ-модель (рекомендуется)":
                with st.spinner("🧠 Генерация ответа с помощью локальной модели..."):
                    result = st.session_state.rag_with_llm.get_answer_local(
                        prompt,
                        model_name=selected_model,
                        use_gpu=use_gpu,
                        quantize=quantize
                    )
            
            else:
                result = st.session_state.rag_with_llm.get_answer_with_openai(prompt)
            
            if "error" in result:
                st.error(f"❌ {result['error']}")
                answer = f"Ошибка: {result['error']}"
                sources = []
            else:
                answer = result["answer"]
                sources = result.get("sources", [])
            
            st.markdown(answer)
            
            if sources:
                with st.expander("📄 Источники информации"):
                    for i, source in enumerate(sources, 1):
                        st.write(f"**{i}. {source['document']}**")
                        st.write(f"Релевантность: {source['score']:.2%}")
                        st.write(f"_{source['text'][:200]}..._")
                        st.divider()
            
            if "model_info" in result:
                st.caption(f"🤖 Модель: {result['model_info']['model_name']} ({result['model_info']['device']})")
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

# === ОЧИСТКА ПРИ ЗАКРЫТИИ ===
# (будет выполнена при перезапуске)