"""
Веб-приложение для RAG-системы с просмотром блоков
"""

import streamlit as st
import os
import tempfile
import time
from rag_system import RAGSystem, RAGWithLLM
from config import MODEL_CONFIG

# === НАСТРОЙКА ВРЕМЕННЫХ ФАЙЛОВ ===
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

def cleanup_old_files():
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

cleanup_old_files()

# === НАСТРОЙКА СТРАНИЦЫ ===
st.set_page_config(
    page_title="RAG-поиск по документам",
    page_icon="📚",
    layout="wide"
)

# === ИНИЦИАЛИЗАЦИЯ ===
if 'rag_system' not in st.session_state:
    with st.spinner("🔄 Загрузка системы..."):
        st.session_state.rag_system = RAGSystem()
        st.session_state.rag_with_llm = RAGWithLLM(st.session_state.rag_system)
        
        with st.spinner("🧠 Загрузка модели... Это может занять 2-5 минут"):
            try:
                result = st.session_state.rag_with_llm.load_local_model(
                    model_name=MODEL_CONFIG["model_name"],
                    use_gpu=MODEL_CONFIG["use_gpu"],
                    max_length=MODEL_CONFIG["max_length"],
                    temperature=MODEL_CONFIG["temperature"]
                )
                if "error" in result:
                    st.error(f"❌ Ошибка загрузки модели: {result['error']}")
                else:
                    st.success("✅ Модель успешно загружена!")
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")

if 'messages' not in st.session_state:
    st.session_state.messages = []

# === ЗАГОЛОВОК ===
st.title("📚 Интеллектуальный помощник по документам")
st.markdown("Загружайте документы и задавайте вопросы")

# === БОКОВАЯ ПАНЕЛЬ ===
with st.sidebar:
    st.header("📂 Управление документами")
    
    uploaded_files = st.file_uploader(
        "Загрузите документы (PDF, DOCX, TXT)",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("📥 Добавить в базу", type="primary", use_container_width=True):
            with st.spinner("Обработка документов..."):
                total_chunks = 0
                failed_files = []
                
                for file in uploaded_files:
                    try:
                        safe_filename = f"{int(time.time() * 1000)}_{file.name}"
                        file_path = os.path.join(TEMP_DIR, safe_filename)
                        
                        with open(file_path, 'wb') as f:
                            f.write(file.getvalue())
                        
                        chunks = st.session_state.rag_system.add_document(file_path)
                        total_chunks += chunks
                        
                        for attempt in range(5):
                            try:
                                if os.path.exists(file_path):
                                    os.unlink(file_path)
                                    break
                            except (PermissionError, OSError):
                                time.sleep(0.3)
                        
                    except Exception as e:
                        failed_files.append(f"{file.name}: {str(e)}")
                
                if failed_files:
                    st.warning(f"⚠️ Ошибки: {', '.join(failed_files)}")
                else:
                    st.success(f"✅ Добавлено {total_chunks} блоков!")
                
                time.sleep(1)
                st.rerun()
    
    st.divider()
    
    # === СТАТИСТИКА ===
    st.header("📊 Статистика")
    
    stats = st.session_state.rag_system.get_database_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="📄 Блоков",
            value=stats["total_chunks"]
        )
    with col2:
        st.metric(
            label="📐 Размерность",
            value=stats["vector_dimension"]
        )
    
    st.divider()
    
    # === ПРОСМОТР БЛОКОВ ===
    st.header("📖 Просмотр блоков")
    
    # Кнопка для показа всех блоков
    if st.button("🔍 Показать все блоки", use_container_width=True):
        st.session_state.show_blocks = True
    
    # Кнопка для скрытия блоков
    if st.button("🙈 Скрыть блоки", use_container_width=True):
        st.session_state.show_blocks = False
    
    st.divider()
    
    # === ОЧИСТКА БАЗЫ ===
    if st.button("🗑️ Очистить базу", type="secondary", use_container_width=True):
        st.session_state.rag_system.clear_database()
        st.success("✅ База очищена!")
        time.sleep(0.5)
        st.rerun()

# === ОСНОВНАЯ ОБЛАСТЬ ===
# Отображение блоков (если включено)
if st.session_state.get('show_blocks', False):
    st.header("📖 Все блоки в базе")
    
    # Получаем все блоки из базы
    try:
        import chromadb
        collection = st.session_state.rag_system.collection
        
        # Получаем все документы
        all_data = collection.get(include=["documents", "metadatas"])
        
        if all_data and all_data['documents']:
            for i, (doc, meta) in enumerate(zip(all_data['documents'], all_data['metadatas'])):
                doc_name = meta.get('document_name', 'Неизвестный')
                block_num = meta.get('block_index', i) + 1
                total = meta.get('total_blocks', '?')
                
                with st.expander(f"📄 Блок {block_num}/{total} из {doc_name}"):
                    # Показываем текст блока
                    st.text_area(
                        label="Текст блока",
                        value=doc[:500] + "..." if len(doc) > 500 else doc,
                        height=150,
                        key=f"block_{i}"
                    )
                    
                    # Показываем информацию о блоке
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"📏 Длина: {len(doc)} симв.")
                    with col2:
                        st.caption(f"📄 Документ: {doc_name}")
                    with col3:
                        st.caption(f"🔢 Номер: {block_num}/{total}")
                    
                    st.divider()
        else:
            st.info("📭 В базе нет блоков. Загрузите документы!")
            
    except Exception as e:
        st.error(f"Ошибка при получении блоков: {e}")

# Чат для вопросов
st.header("💬 Задайте вопрос")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Введите ваш вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🧠 Поиск и генерация ответа..."):
            
            result = st.session_state.rag_with_llm.get_answer_local(prompt)
            
            if "error" in result:
                st.error(f"❌ {result['error']}")
                answer = f"Ошибка: {result['error']}"
            else:
                answer = result["answer"]
            
            st.markdown(answer)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })