from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Ты - ассистент по банковским документам Сбербанка. Отвечай ТОЛЬКО на основе контекста ниже.

ВАЖНЫЕ ПРАВИЛА:
1. Контекст содержит таблицы в формате Markdown (строки с | символами). ЧИТАЙ их внимательно.
2. В таблицах данные расположены по столбцам. Первый столбец - название параметра, остальные - значения по тарифам.
3. Если нашёл нужное значение в таблице - назови его точно, укажи [стр. N].
4. Если информация есть в контексте - ОБЯЗАТЕЛЬНО дай ответ. Не говори "не найдена" если данные есть.
5. Отвечай коротко и по делу, на русском языке.
6. Числа вида 59916, 37417, 59990 где последние 1-2 цифры являются номером сноски - 
   читай как 599, 374, 5999. Сноски в документе обозначаются надстрочными цифрами.
"""


class GigaChatLLM:
    def __init__(self):
        from gigachat import GigaChat

        creds = os.getenv("GIGACHAT_CREDENTIALS")
        scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")

        self._generate_model = os.getenv("GIGACHAT_LLM_MODEL", "GigaChat")
        self._expand_model = os.getenv("GIGACHAT_EXPAND_MODEL", "GigaChat")

        self._client = GigaChat(
            credentials=creds,
            scope=scope,
            verify_ssl_certs=False,
            model=self._generate_model,
        )

        self._expand_client = (
            GigaChat(
                credentials=creds,
                scope=scope,
                verify_ssl_certs=False,
                model=self._expand_model,
            )
            if self._expand_model != self._generate_model
            else self._client
        )

        logger.info(f"GigaChatLLM: generate={self._generate_model} expand={self._expand_model}")

    def expand_query(self, question):
        prompt = (
            "Сгенерируй 2 разных формулировки следующего вопроса для поиска в документе.\n"
            "Каждую формулировку с новой строки, без нумерации, без лишних символов.\n\n"
            f"Вопрос: {question}\n\nФормулировки:"
        )
        try:
            response = self._expand_client.chat(prompt)
            raw = response.choices[0].message.content.strip()
            return [v.strip() for v in raw.split("\n") if v.strip()][:2]
        except Exception as e:
            logger.warning(f"expand_query ошибка: {e}")
            return []

    def generate(self, question, context_chunks):
        context_blocks = []
        for chunk in context_chunks:
            tag = "ТАБЛИЦА" if chunk.is_table else "ТЕКСТ"
            context_blocks.append(f"[{tag} | стр. {chunk.page}]\n{chunk.text}")

        context = "\n\n---\n\n".join(context_blocks)

        logger.info(
            f"Контекст для LLM ({len(context)} симв., {len(context_chunks)} блоков):\n{context[:3000]}"
        )

        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"=== КОНТЕКСТ ИЗ ДОКУМЕНТА ===\n"
            f"{context}\n"
            f"=== КОНЕЦ КОНТЕКСТА ===\n\n"
            f"Вопрос: {question}\n\n"
            f"Ответ (найди информацию в контексте выше и ответь точно):"
        )

        response = self._client.chat(prompt)
        answer = response.choices[0].message.content.strip()
        logger.info(f"Ответ GigaChat: {answer[:300]}")
        return answer