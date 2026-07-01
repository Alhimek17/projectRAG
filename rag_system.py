"""
RAG система с улучшенной сегментацией для больших документов
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
from datetime import datetime
import torch

class TextSplitter:
    """Улучшенный класс для разбивки текста на семантические блоки"""
    
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 300):
        """
        Параметры:
        - chunk_size: 1200 символов для сохранения полных разделов
        - chunk_overlap: 300 символов перекрытия
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """
        Разбивает текст на смысловые блоки с учетом структуры
        """
        # Очистка текста
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) < 50:
            return [text] if text else []
        
        # Пробуем разбить по номерам разделов (1., 2., 3.1, и т.д.)
        section_pattern = r'(?=\d+\.\d+\.\s+[А-ЯA-Z][а-яa-z]+)|(?=\d+\.\s+[А-ЯA-Z][а-яa-z]+)|(?=Раздел\s+\d+)'
        sections = re.split(section_pattern, text)
        
        # Если есть разделы, объединяем их в осмысленные блоки
        if len(sections) > 1:
            merged = []
            current = ""
            for sec in sections:
                sec = sec.strip()
                if not sec:
                    continue
                if len(current) + len(sec) <= self.chunk_size:
                    current += " " + sec if current else sec
                else:
                    if current:
                        merged.append(current.strip())
                    # Начинаем новый блок с перекрытием
                    if self.chunk_overlap > 0 and current:
                        overlap = current[-self.chunk_overlap:] if len(current) > self.chunk_overlap else current
                        current = overlap + " " + sec
                    else:
                        current = sec
            if current:
                merged.append(current.strip())
            return merged if merged else [text[:self.chunk_size]]
        
        # Если нет разделов, разбиваем по предложениям
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
                if self.chunk_overlap > 0 and current_chunk:
                    overlap = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap + " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text[:self.chunk_size]]

class DocumentParser:
    """Класс для парсинга документов"""
    
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
    """Класс для работы с локальной языковой моделью Qwen2.5"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 use_gpu: bool = True,
                 max_length: int = 500,
                 temperature: float = 0.3):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.max_length = max_length
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.device_info = {}
        
        print(f"📥 Подготовка к загрузке модели: {model_name}")
        print(f"💻 Использовать GPU: {use_gpu}")
    
    def check_gpu(self):
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU доступен: {gpu_count} устройств")
            print(f"   Модель GPU: {gpu_name}")
            print(f"   Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠️ GPU не доступен, используется CPU")
            return False
    
    def load_model(self):
        if self.is_loaded:
            print("Модель уже загружена")
            return
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print("⏳ Загрузка модели... Это может занять несколько минут при первом запуске")
            
            gpu_available = self.check_gpu()
            
            if self.use_gpu and gpu_available:
                device = "cuda"
                dtype = torch.float16
                print("🚀 Используется GPU (float16)")
            else:
                device = "cpu"
                dtype = torch.float32
                print("💻 Используется CPU (float32)")
            
            print(f"📥 Загрузка токенизатора {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"📥 Загрузка модели {self.model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            if device == "cpu":
                print("Перемещение модели на CPU...")
                self.model = self.model.to(device)
            
            self.is_loaded = True
            self.device_info = {
                "device": device,
                "dtype": str(dtype),
                "gpu_available": gpu_available,
                "model_size": f"{self.model.num_parameters() / 1e9:.1f}B"
            }
            
            print(f"✅ Модель успешно загружена!")
            print(f"   Устройство: {device}")
            print(f"   Тип данных: {dtype}")
            print(f"   Параметры: {self.model.num_parameters() / 1e9:.1f}B")
            
        except ImportError as e:
            raise ImportError(f"Необходимые библиотеки не установлены: {e}")
        except Exception as e:
            raise Exception(f"Ошибка при загрузке модели: {e}")
    
    def generate_response(self, 
                          context: str, 
                          query: str,
                          max_length: int = None,
                          temperature: float = None) -> str:
        """Генерация ответа на основе контекста"""
        if not self.is_loaded:
            self.load_model()
        
        try:
            import torch
            
            max_new_tokens = max_length or self.max_length
            temp = temperature or self.temperature
            
            # Ограничиваем контекст
            if len(context) > 4000:
                context = context[:4000] + "..."
            
            # Формат промпта для Qwen2.5
            prompt = f"""<|im_start|>system
Ты — эксперт по работе с системой 1С:Документооборот. Отвечай на вопросы пользователя, используя ТОЛЬКО информацию из предоставленного контекста.

Правила:
1. Если в контексте есть ответ на вопрос — дай его четко и структурированно.
2. Если информации недостаточно — скажи: "В предоставленном документе нет информации по этому вопросу."
3. Не выдумывай информацию, которой нет в контексте.
4. При цитировании указывай раздел документа.<|im_end|>
<|im_start|>user
Контекст из документа:

{context}

Вопрос пользователя: {query}

Дай ответ строго на основе контекста.<|im_end|>
<|im_start|>assistant
"""
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                max_length=4096,
                truncation=True
            )
            
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.9
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1].strip()
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0].strip()
            
            return response
            
        except Exception as e:
            return f"Ошибка при генерации ответа: {str(e)}"
    
    def get_model_info(self) -> Dict:
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            **self.device_info
        }

class RAGSystem:
    """Основной класс RAG-системы с улучшенным поиском"""
    
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
        
        self.splitter = TextSplitter(chunk_size=1200, chunk_overlap=300)
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
            print(f"✅ Добавлено {len(chunks)} блоков")
        except Exception as e:
            print(f"Ошибка: {e}")
            return 0
        
        return len(chunks)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.collection.count() == 0:
            return []
        
        query_embedding = self.embedding_model.encode(query).tolist()
        
        try:
            # Ищем в 2 раза больше для лучшего охвата
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 2, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"Ошибка поиска: {e}")
            return []
        
        documents = []
        if results and 'documents' in results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                score = 1 - results['distances'][0][i] if results['distances'] else 0.0
                # Фильтруем только релевантные результаты
                if score > 0.25:
                    documents.append({
                        "text": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "similarity_score": score
                    })
        
        return documents[:top_k]
    
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
    """Класс для RAG с локальной моделью"""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag = rag_system
        self.local_llm = None
    
    def load_local_model(self, 
                         model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", 
                         use_gpu: bool = True,
                         max_length: int = 500,
                         temperature: float = 0.3) -> Dict:
        try:
            self.local_llm = LocalLLM(
                model_name=model_name,
                use_gpu=use_gpu,
                max_length=max_length,
                temperature=temperature
            )
            self.local_llm.load_model()
            return {"success": True, "model_info": self.local_llm.get_model_info()}
        except Exception as e:
            return {"error": str(e)}
    
    def get_answer_local(self, query: str, top_k: int = 3) -> Dict:
        try:
            if self.local_llm is None or not self.local_llm.is_loaded:
                return {"error": "Модель не загружена"}
            
            similar_chunks = self.rag.search(query, top_k=top_k)
            
            if not similar_chunks:
                return {
                    "answer": "📭 В базе документов нет информации по вашему запросу.\n\n💡 Попробуйте:\n• Переформулировать запрос\n• Загрузить больше документов",
                    "sources": []
                }
            
            # Формируем контекст с номерами разделов
            context = ""
            for i, chunk in enumerate(similar_chunks, 1):
                doc_name = chunk['metadata'].get('document_name', 'Неизвестный')
                context += f"\n--- Источник {i} ({doc_name}) ---\n"
                context += chunk['text'] + "\n"
            
            answer = self.local_llm.generate_response(context, query)
            
            return {
                "answer": answer,
                "sources": [
                    {
                        "document": chunk['metadata'].get('document_name', 'Неизвестный'),
                        "text": chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'],
                        "score": chunk['similarity_score']
                    }
                    for chunk in similar_chunks
                ],
                "model_info": self.local_llm.get_model_info()
            }
            
        except Exception as e:
            return {"error": f"Ошибка: {str(e)}"}