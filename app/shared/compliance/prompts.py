"""
Prompts for compliance document analysis using LLMs.

This module contains all prompt templates used in the ComplianceClass
for document classification and title extraction.
"""

SCANNED_DOCUMENT_TITLE_PROMPT = """Ты эксперт по анализу проектной документации Казахстана.

Проанализируй это изображение документа и:
1. Сначала попытайся определить ТИП документа (например: "Пояснительная записка", "Схема", "План", "Смета" и т.д.)
2. Если тип НЕ определен - извлеки НАЗВАНИЕ документа

Верни результат в JSON формате:
{"type": "тип документа или null", "title": "название документа или null"}

Если ничего не найдено, верни {"type": null, "title": null}"""

TEXT_DOCUMENT_TITLE_SYSTEM_PROMPT = """Ты эксперт по анализу проектной документации.
Твоя задача - извлечь основное название документа из текста.
Верни только название документа в виде JSON: {"title": "название документа"}
Если название не найдено, ничего не возвращай"""

TEXT_DOCUMENT_TITLE_USER_PROMPT = """
Извлеки название документа из следующего текста:

{text}

Верни результат в формате JSON."""
"""Prompts for compliance validation tasks"""

SIGNATURE_AND_STAMP_DETECTION_PROMPT = """Analyze this PDF page image carefully.
Does it contain any handwritten signatures or official stamps/seals?

Look for:
1. Handwritten signatures (cursive writing, personal signatures)
2. Digital signature indicators
3. Round or rectangular official stamps
4. Seals with organizational emblems
5. Government or company stamps

Do NOT count:
- Regular images or photos
- Logos in headers/footers
- Graphics or decorative elements
- Text or tables

Respond in JSON format:
{
    "has_signature": true/false,
    "signature_count": number of signatures found on this page,
    "has_stamp": true/false,
    "stamp_count": number of stamps found on this page,
    "stamp_type": "government/company/notary/other/none",
    "confidence": 0.0-1.0
}"""