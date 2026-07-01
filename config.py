"""
Конфигурация системы
"""

# ===== НАСТРОЙКИ МОДЕЛИ =====
MODEL_CONFIG = {
    # Рекомендуемая модель (отличный баланс скорости и качества)
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    
    # Альтернативные модели Qwen (раскомментируйте нужную):
    # "model_name": "Qwen/Qwen2.5-0.5B-Instruct"  # Очень легкая, быстрая
    # "model_name": "Qwen/Qwen2.5-3B-Instruct"    # Мощнее, требует больше памяти
    # "model_name": "Qwen/Qwen-7B-Chat"           # Старая версия 7B
    
    # Использовать GPU (если доступен)
    "use_gpu": True,
    
    # Количество результатов поиска
    "top_k": 3,
    
    # Настройки генерации
    "max_length": 500,
    "temperature": 0.3,
}

# ===== НАСТРОЙКИ БАЗЫ ДАННЫХ =====
DB_CONFIG = {
    "persist_directory": "./chroma_db",
    "collection_name": "normative_documents",
}

# ===== НАСТРОЙКИ ВЕКТОРИЗАЦИИ =====
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "chunk_size": 500,
    "chunk_overlap": 100,
}