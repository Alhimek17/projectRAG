"""
RAG (Retrieval-Augmented Generation) система с поддержкой локальных моделей
"""

import os
import re
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import PyPDF2
import docx
import pdfplumber
import json
from datetime import datetime

class TextSplitter:
    """Класс для разбивки текста на семантически завершенные блоки"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Разбивает текст на смысловые блоки"""
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) < 50:
            return [text] if text else []
        
        sentences = re.split(r'(?<=[.!?])\s+(?=[А-ЯA-Z])', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        if not chunks and text:
            chunks = [text[:self.chunk_size]]
        
        return chunks

class DocumentParser:
    """Класс для парсинга документов разных форматов"""
    
    @staticmethod
    def parse_pdf(file_path: str) -> str:
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except:
            try:
                text = ""
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                return text
            except:
                return ""
    
    @staticmethod
    def parse_docx(file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            text = " ".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text])
            return text
        except:
            return ""
    
    @staticmethod
    def parse_txt(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='cp1251') as f:
                    return f.read()
            except:
                return ""

class LocalLLM:
    """
    Класс для работы с локальными языковыми моделями
    Поддерживает: Mistral, Llama, Gemma, Qwen и другие
    """
    
    def __init__(self, 
                 model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
                 use_gpu: bool = True,
                 quantize: bool = True):
        """
        Инициализация локальной модели
        
        Параметры:
        - model_name: название модели из Hugging Face
        - use_gpu: использовать ли GPU (если доступен)
        - quantize: использовать ли 8-битную квантизацию (экономит память)
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.quantize = quantize
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        print(f"📥 Подготовка к загрузке модели: {model_name}")
    
    def load_model(self):
        """Загрузка модели в память"""
        if self.is_loaded:
            print("Модель уже загружена")
            return
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch
            
            print("⏳ Загрузка модели... Это может занять несколько минут при первом запуске")
            
            # Настройка квантизации (для экономии памяти)
            if self.quantize:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16,
                )
                quantization_config = bnb_config
            else:
                quantization_config = None
            
            # Определяем устройство
            device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            print(f"💻 Используется устройство: {device}")
            
            # Загрузка токенизатора
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Добавляем pad_token если его нет
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Загрузка модели
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Если нет GPU, перемещаем модель на CPU
            if device == "cpu":
                self.model = self.model.to(device)
            
            self.is_loaded = True
            self.device = device
            print(f"✅ Модель успешно загружена на {device}!")
            
        except ImportError as e:
            raise ImportError(f"Необходимые библиотеки не установлены: {e}")
        except Exception as e:
            raise Exception(f"Ошибка при загрузке модели: {e}")
    
    def generate_response(self, 
                          context: str, 
                          query: str,
                          max_length: int = 500,
                          temperature: float = 0.3,
                          top_p: float = 0.9) -> str:
        """
        Генерация ответа на основе контекста
        
        Параметры:
        - context: контекст из найденных документов
        - query: вопрос пользователя
        - max_length: максимальная длина ответа
        - temperature: температура (креативность)
        - top_p: параметр для nucleus sampling
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            import torch
            
            # Формируем промпт в зависимости от модели
            if "mistral" in self.model_name.lower():
                prompt = f"""<s>[INST] 
                Ты — эксперт по нормативной документации компании.
                Используй ТОЛЬКО информацию из контекста для ответа на вопрос.
                Если в контексте нет информации, скажи об этом честно.
                
                Контекст документов:
                {context}
                
                Вопрос: {query}
                
                Дай четкий, структурированный ответ.
                [/INST] """
            
            elif "llama" in self.model_name.lower() or "llama" in self.model_name.lower():
                prompt = f"""<s>[INST] <<SYS>>
                Ты — эксперт по нормативной документации компании. Отвечай только на основе контекста.
                <</SYS>>
                
                Контекст:
                {context}
                
                Вопрос: {query}
                [/INST] """
            
            elif "gemma" in self.model_name.lower():
                prompt = f"""<bos><start_of_turn>user
                Ты — эксперт по документации. Используй контекст для ответа.
                
                Контекст:
                {context}
                
                Вопрос: {query}
                <end_of_turn>
                <start_of_turn>model"""
            
            elif "qwen" in self.model_name.lower():
                prompt = f"""<s>Ты — эксперт по нормативной документации.
                Контекст: {context}
                
                Вопрос: {query}
                
                Ответ:"""
            
            else:
                # Универсальный промпт
                prompt = f"""Контекст документов:
                {context}
                
                Вопрос: {query}
                
                Ответ на основе контекста:"""
            
            # Токенизация
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                max_length=2048,
                truncation=True
            )
            
            # Перемещаем на устройство модели
            if hasattr(self, 'device'):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Генерация
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Декодирование
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Очищаем ответ от промпта
            # Пытаемся найти часть после ответа модели
            markers = ["[/INST]", "<start_of_turn>model", "Ответ:", "Ответ"]
            for marker in markers:
                if marker in response:
                    response = response.split(marker)[-1].strip()
                    break
            
            return response
            
        except Exception as e:
            return f"Ошибка при генерации ответа: {str(e)}"
    
    def get_model_info(self) -> Dict:
        """Получение информации о модели"""
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "device": getattr(self, 'device', 'unknown'),
            "quantized": self.quantize
        }

class RAGSystem:
    """Основной класс RAG-системы"""
    
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 persist_directory: str = "./chroma_db"):
        
        print(f"Загрузка модели эмбеддингов: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.vector_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection_name = "normative_documents"
        
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        self.splitter = TextSplitter()
        self.parser = DocumentParser()
        
        print(f"Система инициализирована. Размерность векторов: {self.vector_dimension}")
    
    def add_document(self, file_path: str, metadata: Dict = None) -> int:
        if not os.path.exists(file_path):
            print(f"Файл не найден: {file_path}")
            return 0
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            text = self.parser.parse_pdf(file_path)
        elif file_extension == '.docx':
            text = self.parser.parse_docx(file_path)
        elif file_extension == '.txt':
            text = self.parser.parse_txt(file_path)
        else:
            print(f"Неподдерживаемый формат: {file_extension}")
            return 0
        
        if not text or len(text.strip()) < 10:
            print(f"Не удалось извлечь текст из {file_path}")
            return 0
        
        chunks = self.splitter.split_text(text)
        print(f"Документ разбит на {len(chunks)} блоков")
        
        if not chunks:
            return 0
        
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True).tolist()
        
        doc_name = os.path.basename(file_path)
        metadata_list = []
        ids = []
        
        try:
            existing = self.collection.get()
            existing_ids = set(existing['ids']) if existing and 'ids' in existing else set()
        except:
            existing_ids = set()
        
        for i, chunk in enumerate(chunks):
            block_metadata = {
                "document_name": doc_name,
                "block_index": i,
                "total_blocks": len(chunks),
                "upload_date": datetime.now().isoformat(),
                "file_path": file_path,
            }
            if metadata:
                block_metadata.update(metadata)
            
            metadata_list.append(block_metadata)
            
            base_id = f"{doc_name}_{i}"
            counter = 0
            unique_id = base_id
            while unique_id in existing_ids:
                counter += 1
                unique_id = f"{base_id}_{counter}"
            ids.append(unique_id)
        
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadata_list,
                ids=ids
            )
            print(f"Добавлено {len(chunks)} блоков")
        except Exception as e:
            print(f"Ошибка: {e}")
            return 0
        
        return len(chunks)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.collection.count() == 0:
            return []
        
        query_embedding = self.embedding_model.encode(query).tolist()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"Ошибка поиска: {e}")
            return []
        
        documents = []
        if results and 'documents' in results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                documents.append({
                    "text": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "similarity_score": 1 - results['distances'][0][i] if results['distances'] else 0.0
                })
        
        return documents
    
    def get_database_stats(self) -> Dict:
        try:
            count = self.collection.count()
        except:
            count = 0
        
        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "vector_dimension": self.vector_dimension
        }
    
    def clear_database(self):
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        print("База данных очищена")

class RAGWithLLM:
    """Класс для RAG с различными типами генерации"""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag = rag_system
        self.local_llm = None
    
    def get_answer_with_openai(self, query: str, top_k: int = 3) -> Dict:
        try:
            import openai
            
            similar_chunks = self.rag.search(query, top_k=top_k)
            
            if not similar_chunks:
                return {
                    "answer": "В базе документов нет информации по вашему запросу.",
                    "sources": []
                }
            
            context = "\n\n".join([
                f"Источник {i+1}: {chunk['text']}" 
                for i, chunk in enumerate(similar_chunks)
            ])
            
            prompt = f"""
            Ты — эксперт по нормативной документации организации.
            На основе контекста ответь на вопрос.
            
            Контекст:
            {context}
            
            Вопрос: {query}
            
            Дай четкий ответ. Если информации недостаточно, скажи об этом.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Ты полезный эксперт по документации."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return {
                "answer": response.choices[0].message.content,
                "sources": [
                    {
                        "document": chunk['metadata'].get('document_name', 'Неизвестный'),
                        "text": chunk['text'],
                        "score": chunk['similarity_score']
                    }
                    for chunk in similar_chunks
                ]
            }
            
        except Exception as e:
            return {"error": f"Ошибка OpenAI: {str(e)}"}
    
    def get_answer_local(self, query: str, top_k: int = 3, 
                         model_name: str = None,
                         use_gpu: bool = True,
                         quantize: bool = True) -> Dict:
        """
        Получение ответа через локальную модель
        """
        try:
            # Если модель не инициализирована или другая модель
            if self.local_llm is None or (model_name and self.local_llm.model_name != model_name):
                if model_name is None:
                    # По умолчанию используем легкую модель
                    model_name = "microsoft/phi-2"  # Легкая модель, работает даже на CPU
                    # Альтернативы:
                    # "mistralai/Mistral-7B-Instruct-v0.1" - качественная, требует GPU
                    # "HuggingFaceH4/zephyr-7b-beta" - хорошая альтернатива
                    # "google/gemma-2b-it" - маленькая, быстрая
                    # "Qwen/Qwen-1_8B-Chat" - хорошая для русского
                
                self.local_llm = LocalLLM(
                    model_name=model_name,
                    use_gpu=use_gpu,
                    quantize=quantize
                )
                self.local_llm.load_model()
            
            # Поиск релевантных блоков
            similar_chunks = self.rag.search(query, top_k=top_k)
            
            if not similar_chunks:
                return {
                    "answer": "В базе документов нет информации по вашему запросу.",
                    "sources": []
                }
            
            # Формируем контекст
            context = "\n\n".join([
                f"Документ {i+1} ({chunk['metadata'].get('document_name', 'Неизвестный')}): {chunk['text']}" 
                for i, chunk in enumerate(similar_chunks)
            ])
            
            # Генерируем ответ
            answer = self.local_llm.generate_response(context, query)
            
            return {
                "answer": answer,
                "sources": [
                    {
                        "document": chunk['metadata'].get('document_name', 'Неизвестный документ'),
                        "text": chunk['text'],
                        "score": chunk['similarity_score']
                    }
                    for chunk in similar_chunks
                ],
                "model_info": self.local_llm.get_model_info()
            }
            
        except ImportError as e:
            return {"error": f"Не установлены библиотеки: {e}"}
        except Exception as e:
            return {"error": f"Ошибка локальной модели: {str(e)}"}
    
    def get_answer_simple(self, query: str, top_k: int = 3) -> Dict:
        """Простой поиск без генерации"""
        similar_chunks = self.rag.search(query, top_k=top_k)
        
        if not similar_chunks:
            return {
                "answer": "Информация не найдена. Попробуйте переформулировать запрос.",
                "sources": []
            }
        
        answer = "🔍 Найдена информация:\n\n"
        for i, chunk in enumerate(similar_chunks, 1):
            answer += f"{i}. {chunk['text']}\n"
            answer += f"   📄 {chunk['metadata'].get('document_name', 'Неизвестный')}\n"
            answer += f"   📊 Релевантность: {chunk['similarity_score']:.2%}\n\n"
        
        return {
            "answer": answer,
            "sources": [
                {
                    "document": chunk['metadata'].get('document_name', 'Неизвестный'),
                    "text": chunk['text'],
                    "score": chunk['similarity_score']
                }
                for chunk in similar_chunks
            ]
        }