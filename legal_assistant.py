# pip install python-docx pdfplumber sentence-transformers faiss-cpu gigachat python-telegram-bot
import os
import re
import hashlib
from typing import Dict, List, Optional
from telegram import Update, Document
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pdfplumber
from docx import Document as DocxDocument

# --- Константы и промпт ---
BLACKLIST = ["sudo", "rm -rf", "перезагрузи", "debug", "забудь всё", "ignore your prompt", "забудь инструкции"]
ALLOWED_EXTENSIONS = {'doc', 'docx', 'pdf', 'txt'}
MAX_FILE_SIZE = 10 * 1024 * 1024

LEGAL_PROMPT = """Ты — юрист, анализирующий договор купли-продажи 100% доли в уставном капитале ООО на предмет рисков со стороны Покупателя.
Анализ проводится по российскому праву, используй только актуальную нормативную базу и судебную практику.  
Контекст:
- Договор купли-продажи между ООО "Ландыш" (Покупатель) и ООО "Тюльпан" (Продавец).
- Предмет договора: 100% доли в уставном капитале ООО "Тюльпан", номинальной стоимости 100 миллионов рублей.
- Срок оплаты цены по договору: с 01.06.2025 по 10.06.2025.
- В случае неуплаты цены по договору возникает залог в силу закона.

Задача:
Выявить риски для Покупателя, включая:
  - Финансовые риски (штрафы, неустойки и прочее).
  - Юридические риски (неопределенные формулировки, отсутствие существенных условий).

Формат вывода:
Представь результаты в виде таблицы со следующими колонками:
1. Пункт договора
2. Описание риска
3. Обоснование вывода о риске
4. Рекомендация по устранению риска"""

# --- RAG с корпоративным правом ---
CORPORATE_LAW_TEXTS = [
    "Статья 21 ФЗ 'Об ООО': Уступка доли в уставном капитале требует нотариального удостоверения.",
    "Ст. 22.1 ФЗ 'Об ООО': Залог доли подлежит внесению в ЕГРЮЛ.",
    "П. 3 ст. 23 ФЗ 'Об ООО': Продажа доли третьим лицам может быть ограничена уставом.",
    "Постановление Пленума ВАС №19: Споры о качестве передаваемой доли рассматриваются с учетом действительной стоимости.",
    "Статья 488 ГК РФ: Последствия нарушения условий о платеже за товар."
]

# --- Инициализация моделей ---
rag_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
giga = GigaChat(credentials=os.getenv("GIGACHAT_CREDENTIALS"), verify_ssl_certs=False)

class RAGSystem:
    def __init__(self):
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        self.documents = []
        self._build_index()
    
    def _build_index(self):
        texts = CORPORATE_LAW_TEXTS
        embeddings = rag_model.encode(texts, normalize_embeddings=True)
        self.index.add_with_ids(embeddings, np.arange(len(texts)))
        self.documents = texts

    def search(self, query: str, k=3, threshold=0.7) -> List[str]:
        query_embed = rag_model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_embed, k)
        return [self.documents[i] for i, d in zip(indices[0], distances[0]) if d > threshold]

# --- Система памяти ---
class Memory:
    def __init__(self):
        self.storage: Dict[int, List[Dict]] = {}
    
    def add(self, chat_id: int, role: str, content: str):
        if chat_id not in self.storage:
            self.storage[chat_id] = []
        self.storage[chat_id].append({"role": role, "content": content})

# --- Парсер документов ---
class DocumentParser:
    @staticmethod
    def parse(file_bytes: bytes, extension: str) -> Optional[str]:
        try:
            if extension == 'pdf':
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    return '\n'.join([page.extract_text() for page in pdf.pages])
            
            elif extension == 'docx':
                return '\n'.join([p.text for p in DocxDocument(io.BytesIO(file_bytes)).paragraphs])
            
            elif extension == 'txt':
                return file_bytes.decode()
            
        except Exception as e:
            print(f"Ошибка парсинга: {str(e)}")
            return None

# --- Обработчик ошибок ---
class ErrorHandler:
    @staticmethod
    def handle(error: Exception) -> str:
        if "timed out" in str(error):
            return "Ошибка: превышено время ожидания ответа"
        return "Произошла ошибка при обработке запроса"

# --- Основной бот ---
class LegalAssistantBot:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.rag = RAGSystem()
        self.memory = Memory()
        self.parser = DocumentParser()
        self.app = Application.builder().token(self.token).build()
        
        self.app.add_handler(MessageHandler(
            filters.TEXT | filters.Document.ALL, self._handle_message))

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            chat_id = update.effective_chat.id
            
            if update.message.document:
                await self._process_document(update)
            else:
                await self._process_text(update, chat_id)
        
        except Exception as e:
            await update.message.reply_text(ErrorHandler.handle(e))

    async def _process_document(self, update: Update):
        doc = update.message.document
        if doc.file_size > MAX_FILE_SIZE:
            raise ValueError("Файл слишком большой")
        
        ext = doc.file_name.split('.')[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError("Недопустимый формат файла")
        
        file = await doc.get_file()
        content = await file.download_as_bytearray()
        text = self.parser.parse(content, ext)
        
        if not text:
            raise ValueError("Не удалось распарсить документ")
        
        analysis = self._analyze_contract(text)
        await update.message.reply_text(analysis)

    def _analyze_contract(self, text: str) -> str:
        context = "\n".join(self.rag.search(text[:500]))
        
        try:
            response = giga.chat(Chat(
                messages=[
                    Messages(role=MessagesRole.SYSTEM, content=LEGAL_PROMPT),
                    Messages(role=MessagesRole.USER, content=f"Договор:\n{text}\n\nАнализ:")
                ])).choices[0].message.content
            
            if self._check_confidence(response):
                return Critic.validate(response)
            return "Не удалось сформировать уверенный ответ"
        
        except Exception as e:
            return ErrorHandler.handle(e)

    def _check_confidence(self, response: str) -> bool:
        check_prompt = f"""Оцени уверенность ответа от 0 до 1:
        Ответ: {response}
        Только числовой рейтинг без пояснений:"""
        
        score = float(giga.chat(Chat(
            messages=[Messages(role=MessagesRole.USER, content=check_prompt)]
        ).choices[0].message.content)
        
        return score > 0.7

class Critic:
    @staticmethod
    def validate(response: str) -> str:
        critique = giga.chat(Chat(messages=[
            Messages(role=MessagesRole.SYSTEM, content="Выяви пропущенные риски и ошибки в анализе"),
            Messages(role=MessagesRole.USER, content=response)
        ])).choices[0].message.content
        
        if "критических ошибок не найдено" in critique.lower():
            return response
        else:
            return f"Критика:\n{critique}\n\nИсправленный анализ:\n{response}"

if __name__ == "__main__":
    bot = LegalAssistantBot()
    bot.app.run_polling()