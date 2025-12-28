"""Extended Language Catalogs.

This module provides complete message catalogs for additional languages:
- Portuguese (pt) - with Brazilian and European variants
- Italian (it)
- Russian (ru)

These catalogs are complete translations of all validator messages,
ready for enterprise deployment.

Usage:
    from truthound.validators.i18n.extended_catalogs import (
        get_portuguese_messages,
        get_italian_messages,
        get_russian_messages,
        get_arabic_messages,
        get_hebrew_messages,
    )

    # Get Russian catalog
    catalog = get_russian_messages()
    print(catalog.get("null.values_found"))
    # -> "Обнаружено {count} пустых значений в столбце '{column}'"
"""

from __future__ import annotations

from truthound.validators.i18n.catalogs import ValidatorMessageCatalog


# ==============================================================================
# Portuguese Messages (Complete)
# ==============================================================================

_PORTUGUESE_MESSAGES: dict[str, str] = {
    # Null/Completeness
    "null.values_found": "Encontrados {count} valores nulos na coluna '{column}'",
    "null.column_empty": "A coluna '{column}' está completamente vazia",
    "null.above_threshold": "A taxa de nulos ({ratio:.1%}) excede o limite ({threshold:.1%}) na coluna '{column}'",

    # Uniqueness
    "unique.duplicates_found": "Encontrados {count} valores duplicados na coluna '{column}'",
    "unique.composite_duplicates": "Encontradas {count} combinações duplicadas para as colunas {columns}",
    "unique.key_violation": "Violação de chave primária: {count} chaves duplicadas em '{column}'",

    # Type
    "type.mismatch": "Incompatibilidade de tipo na coluna '{column}': esperado {expected}, encontrado {actual}",
    "type.coercion_failed": "Não foi possível converter {count} valores na coluna '{column}' para {target_type}",
    "type.inference_failed": "Não foi possível inferir o tipo da coluna '{column}'",

    # Format
    "format.invalid_email": "Encontrados {count} endereços de email inválidos na coluna '{column}'",
    "format.invalid_phone": "Encontrados {count} números de telefone inválidos na coluna '{column}'",
    "format.invalid_date": "Encontradas {count} datas inválidas na coluna '{column}'",
    "format.invalid_url": "Encontrados {count} URLs inválidos na coluna '{column}'",
    "format.pattern_mismatch": "Encontrados {count} valores que não correspondem ao padrão '{pattern}' na coluna '{column}'",
    "format.regex_failed": "Validação de regex falhou para a coluna '{column}': {count} valores não correspondentes",

    # Range
    "range.out_of_bounds": "Encontrados {count} valores fora do intervalo [{min}, {max}] na coluna '{column}'",
    "range.below_minimum": "Encontrados {count} valores abaixo do mínimo ({min}) na coluna '{column}'",
    "range.above_maximum": "Encontrados {count} valores acima do máximo ({max}) na coluna '{column}'",
    "range.outlier_detected": "Detectados {count} outliers estatísticos na coluna '{column}'",

    # Referential
    "ref.foreign_key_violation": "Encontradas {count} violações de chave estrangeira na coluna '{column}'",
    "ref.missing_reference": "Encontrados {count} valores em '{column}' não presentes na coluna de referência '{ref_column}'",
    "ref.orphan_records": "Encontrados {count} registros órfãos sem correspondência na coluna '{column}'",

    # Statistical
    "stat.distribution_anomaly": "Anomalia de distribuição detectada na coluna '{column}': {details}",
    "stat.mean_out_of_range": "O valor médio ({mean:.2f}) está fora do intervalo esperado [{min}, {max}] para a coluna '{column}'",
    "stat.variance_anomaly": "Variância incomum ({variance:.2f}) detectada na coluna '{column}'",
    "stat.skewness_anomaly": "Assimetria ({skewness:.2f}) indica distribuição não normal na coluna '{column}'",

    # Schema
    "schema.column_missing": "A coluna esperada '{column}' está faltando no conjunto de dados",
    "schema.column_extra": "Coluna inesperada '{column}' encontrada no conjunto de dados",
    "schema.type_mismatch": "A coluna '{column}' tem tipo {actual}, esperado {expected}",
    "schema.constraint_violated": "Restrição '{constraint}' violada para a coluna '{column}'",

    # Cross-table
    "cross.column_mismatch": "Valores na coluna '{column1}' não correspondem à coluna '{column2}'",
    "cross.consistency_failed": "Verificação de consistência entre colunas falhou: {details}",

    # Timeout
    "timeout.exceeded": "A validação expirou após {seconds}s para '{operation}'",
    "timeout.partial_result": "Resultados parciais retornados devido ao tempo limite: {validated}% dos dados validados",

    # General
    "validation.failed": "Validação falhou para a coluna '{column}': {reason}",
    "validation.skipped": "Validação ignorada para a coluna '{column}': {reason}",
    "validation.error": "Erro durante a validação: {error}",
}


# Brazilian Portuguese specific overrides
_PORTUGUESE_BR_MESSAGES: dict[str, str] = {
    "format.invalid_phone": "Encontrados {count} números de celular inválidos na coluna '{column}'",
    "null.column_empty": "A coluna '{column}' tá completamente vazia",
}


# European Portuguese specific overrides
_PORTUGUESE_PT_MESSAGES: dict[str, str] = {
    "format.invalid_phone": "Encontrados {count} números de telemóvel inválidos na coluna '{column}'",
}


# ==============================================================================
# Italian Messages (Complete)
# ==============================================================================

_ITALIAN_MESSAGES: dict[str, str] = {
    # Null/Completeness
    "null.values_found": "Trovati {count} valori nulli nella colonna '{column}'",
    "null.column_empty": "La colonna '{column}' è completamente vuota",
    "null.above_threshold": "Il rapporto nulli ({ratio:.1%}) supera la soglia ({threshold:.1%}) nella colonna '{column}'",

    # Uniqueness
    "unique.duplicates_found": "Trovati {count} valori duplicati nella colonna '{column}'",
    "unique.composite_duplicates": "Trovate {count} combinazioni duplicate per le colonne {columns}",
    "unique.key_violation": "Violazione chiave primaria: {count} chiavi duplicate in '{column}'",

    # Type
    "type.mismatch": "Tipo non corrispondente nella colonna '{column}': previsto {expected}, trovato {actual}",
    "type.coercion_failed": "Impossibile convertire {count} valori nella colonna '{column}' in {target_type}",
    "type.inference_failed": "Impossibile dedurre il tipo della colonna '{column}'",

    # Format
    "format.invalid_email": "Trovati {count} indirizzi email non validi nella colonna '{column}'",
    "format.invalid_phone": "Trovati {count} numeri di telefono non validi nella colonna '{column}'",
    "format.invalid_date": "Trovate {count} date non valide nella colonna '{column}'",
    "format.invalid_url": "Trovati {count} URL non validi nella colonna '{column}'",
    "format.pattern_mismatch": "Trovati {count} valori che non corrispondono al pattern '{pattern}' nella colonna '{column}'",
    "format.regex_failed": "Validazione regex fallita per la colonna '{column}': {count} valori non corrispondenti",

    # Range
    "range.out_of_bounds": "Trovati {count} valori fuori dall'intervallo [{min}, {max}] nella colonna '{column}'",
    "range.below_minimum": "Trovati {count} valori sotto il minimo ({min}) nella colonna '{column}'",
    "range.above_maximum": "Trovati {count} valori sopra il massimo ({max}) nella colonna '{column}'",
    "range.outlier_detected": "Rilevati {count} outlier statistici nella colonna '{column}'",

    # Referential
    "ref.foreign_key_violation": "Trovate {count} violazioni di chiave esterna nella colonna '{column}'",
    "ref.missing_reference": "Trovati {count} valori in '{column}' non presenti nella colonna di riferimento '{ref_column}'",
    "ref.orphan_records": "Trovati {count} record orfani senza corrispondenza in '{column}'",

    # Statistical
    "stat.distribution_anomaly": "Anomalia di distribuzione rilevata nella colonna '{column}': {details}",
    "stat.mean_out_of_range": "Il valore medio ({mean:.2f}) è fuori dall'intervallo previsto [{min}, {max}] per la colonna '{column}'",
    "stat.variance_anomaly": "Varianza insolita ({variance:.2f}) rilevata nella colonna '{column}'",
    "stat.skewness_anomaly": "L'asimmetria ({skewness:.2f}) indica una distribuzione non normale nella colonna '{column}'",

    # Schema
    "schema.column_missing": "La colonna prevista '{column}' manca nel dataset",
    "schema.column_extra": "Colonna inaspettata '{column}' trovata nel dataset",
    "schema.type_mismatch": "La colonna '{column}' ha tipo {actual}, previsto {expected}",
    "schema.constraint_violated": "Vincolo '{constraint}' violato per la colonna '{column}'",

    # Cross-table
    "cross.column_mismatch": "I valori nella colonna '{column1}' non corrispondono alla colonna '{column2}'",
    "cross.consistency_failed": "Controllo di coerenza tra colonne fallito: {details}",

    # Timeout
    "timeout.exceeded": "La validazione è scaduta dopo {seconds}s per '{operation}'",
    "timeout.partial_result": "Risultati parziali restituiti a causa del timeout: {validated}% dei dati validati",

    # General
    "validation.failed": "Validazione fallita per la colonna '{column}': {reason}",
    "validation.skipped": "Validazione saltata per la colonna '{column}': {reason}",
    "validation.error": "Errore durante la validazione: {error}",
}


# ==============================================================================
# Russian Messages (Complete)
# ==============================================================================

_RUSSIAN_MESSAGES: dict[str, str] = {
    # Null/Completeness
    "null.values_found": "Обнаружено {count} пустых значений в столбце '{column}'",
    "null.column_empty": "Столбец '{column}' полностью пуст",
    "null.above_threshold": "Доля пустых значений ({ratio:.1%}) превышает порог ({threshold:.1%}) в столбце '{column}'",

    # Uniqueness
    "unique.duplicates_found": "Обнаружено {count} дублирующихся значений в столбце '{column}'",
    "unique.composite_duplicates": "Обнаружено {count} дублирующихся комбинаций для столбцов {columns}",
    "unique.key_violation": "Нарушение первичного ключа: {count} дублирующихся ключей в '{column}'",

    # Type
    "type.mismatch": "Несоответствие типа в столбце '{column}': ожидался {expected}, найден {actual}",
    "type.coercion_failed": "Не удалось преобразовать {count} значений в столбце '{column}' в {target_type}",
    "type.inference_failed": "Не удалось определить тип столбца '{column}'",

    # Format
    "format.invalid_email": "Обнаружено {count} недействительных email-адресов в столбце '{column}'",
    "format.invalid_phone": "Обнаружено {count} недействительных номеров телефона в столбце '{column}'",
    "format.invalid_date": "Обнаружено {count} недействительных дат в столбце '{column}'",
    "format.invalid_url": "Обнаружено {count} недействительных URL в столбце '{column}'",
    "format.pattern_mismatch": "Обнаружено {count} значений, не соответствующих шаблону '{pattern}' в столбце '{column}'",
    "format.regex_failed": "Проверка регулярного выражения не прошла для столбца '{column}': {count} несоответствующих значений",

    # Range
    "range.out_of_bounds": "Обнаружено {count} значений вне диапазона [{min}, {max}] в столбце '{column}'",
    "range.below_minimum": "Обнаружено {count} значений ниже минимума ({min}) в столбце '{column}'",
    "range.above_maximum": "Обнаружено {count} значений выше максимума ({max}) в столбце '{column}'",
    "range.outlier_detected": "Обнаружено {count} статистических выбросов в столбце '{column}'",

    # Referential
    "ref.foreign_key_violation": "Обнаружено {count} нарушений внешнего ключа в столбце '{column}'",
    "ref.missing_reference": "Обнаружено {count} значений в '{column}', отсутствующих в ссылочном столбце '{ref_column}'",
    "ref.orphan_records": "Обнаружено {count} сиротских записей без соответствия в '{column}'",

    # Statistical
    "stat.distribution_anomaly": "Обнаружена аномалия распределения в столбце '{column}': {details}",
    "stat.mean_out_of_range": "Среднее значение ({mean:.2f}) выходит за ожидаемый диапазон [{min}, {max}] для столбца '{column}'",
    "stat.variance_anomaly": "Необычная дисперсия ({variance:.2f}) обнаружена в столбце '{column}'",
    "stat.skewness_anomaly": "Асимметрия ({skewness:.2f}) указывает на ненормальное распределение в столбце '{column}'",

    # Schema
    "schema.column_missing": "Ожидаемый столбец '{column}' отсутствует в наборе данных",
    "schema.column_extra": "Неожиданный столбец '{column}' найден в наборе данных",
    "schema.type_mismatch": "Столбец '{column}' имеет тип {actual}, ожидался {expected}",
    "schema.constraint_violated": "Ограничение '{constraint}' нарушено для столбца '{column}'",

    # Cross-table
    "cross.column_mismatch": "Значения в столбце '{column1}' не соответствуют столбцу '{column2}'",
    "cross.consistency_failed": "Проверка согласованности между столбцами не прошла: {details}",

    # Timeout
    "timeout.exceeded": "Время валидации истекло после {seconds}с для '{operation}'",
    "timeout.partial_result": "Частичные результаты возвращены из-за тайм-аута: {validated}% данных проверено",

    # General
    "validation.failed": "Валидация не прошла для столбца '{column}': {reason}",
    "validation.skipped": "Валидация пропущена для столбца '{column}': {reason}",
    "validation.error": "Ошибка во время валидации: {error}",
}


# ==============================================================================
# Arabic Messages (Complete)
# ==============================================================================

_ARABIC_MESSAGES: dict[str, str] = {
    # Null/Completeness
    "null.values_found": "تم العثور على {count} قيمة فارغة في العمود '{column}'",
    "null.column_empty": "العمود '{column}' فارغ تماماً",
    "null.above_threshold": "نسبة القيم الفارغة ({ratio:.1%}) تتجاوز الحد ({threshold:.1%}) في العمود '{column}'",

    # Uniqueness
    "unique.duplicates_found": "تم العثور على {count} قيمة مكررة في العمود '{column}'",
    "unique.composite_duplicates": "تم العثور على {count} تركيبة مكررة للأعمدة {columns}",
    "unique.key_violation": "انتهاك المفتاح الأساسي: {count} مفتاح مكرر في '{column}'",

    # Type
    "type.mismatch": "عدم تطابق النوع في العمود '{column}': متوقع {expected}، وُجد {actual}",
    "type.coercion_failed": "تعذر تحويل {count} قيمة في العمود '{column}' إلى {target_type}",
    "type.inference_failed": "تعذر استنتاج نوع العمود '{column}'",

    # Format
    "format.invalid_email": "تم العثور على {count} عنوان بريد إلكتروني غير صالح في العمود '{column}'",
    "format.invalid_phone": "تم العثور على {count} رقم هاتف غير صالح في العمود '{column}'",
    "format.invalid_date": "تم العثور على {count} تاريخ غير صالح في العمود '{column}'",
    "format.invalid_url": "تم العثور على {count} عنوان URL غير صالح في العمود '{column}'",
    "format.pattern_mismatch": "تم العثور على {count} قيمة لا تطابق النمط '{pattern}' في العمود '{column}'",
    "format.regex_failed": "فشل التحقق من التعبير النمطي للعمود '{column}': {count} قيمة غير مطابقة",

    # Range
    "range.out_of_bounds": "تم العثور على {count} قيمة خارج النطاق [{min}, {max}] في العمود '{column}'",
    "range.below_minimum": "تم العثور على {count} قيمة أقل من الحد الأدنى ({min}) في العمود '{column}'",
    "range.above_maximum": "تم العثور على {count} قيمة أعلى من الحد الأقصى ({max}) في العمود '{column}'",
    "range.outlier_detected": "تم اكتشاف {count} قيمة شاذة إحصائياً في العمود '{column}'",

    # Referential
    "ref.foreign_key_violation": "تم العثور على {count} انتهاك للمفتاح الخارجي في العمود '{column}'",
    "ref.missing_reference": "تم العثور على {count} قيمة في '{column}' غير موجودة في العمود المرجعي '{ref_column}'",
    "ref.orphan_records": "تم العثور على {count} سجل يتيم بدون تطابق في '{column}'",

    # Statistical
    "stat.distribution_anomaly": "تم اكتشاف شذوذ في التوزيع في العمود '{column}': {details}",
    "stat.mean_out_of_range": "القيمة المتوسطة ({mean:.2f}) خارج النطاق المتوقع [{min}, {max}] للعمود '{column}'",
    "stat.variance_anomaly": "تباين غير عادي ({variance:.2f}) تم اكتشافه في العمود '{column}'",
    "stat.skewness_anomaly": "الالتواء ({skewness:.2f}) يشير إلى توزيع غير طبيعي في العمود '{column}'",

    # Schema
    "schema.column_missing": "العمود المتوقع '{column}' مفقود من مجموعة البيانات",
    "schema.column_extra": "تم العثور على عمود غير متوقع '{column}' في مجموعة البيانات",
    "schema.type_mismatch": "العمود '{column}' له النوع {actual}، المتوقع {expected}",
    "schema.constraint_violated": "تم انتهاك القيد '{constraint}' للعمود '{column}'",

    # Cross-table
    "cross.column_mismatch": "القيم في العمود '{column1}' لا تتطابق مع العمود '{column2}'",
    "cross.consistency_failed": "فشل التحقق من الاتساق بين الأعمدة: {details}",

    # Timeout
    "timeout.exceeded": "انتهت مهلة التحقق بعد {seconds} ثانية لـ '{operation}'",
    "timeout.partial_result": "تم إرجاع نتائج جزئية بسبب انتهاء المهلة: تم التحقق من {validated}% من البيانات",

    # General
    "validation.failed": "فشل التحقق للعمود '{column}': {reason}",
    "validation.skipped": "تم تخطي التحقق للعمود '{column}': {reason}",
    "validation.error": "خطأ أثناء التحقق: {error}",
}


# ==============================================================================
# Hebrew Messages (Complete)
# ==============================================================================

_HEBREW_MESSAGES: dict[str, str] = {
    # Null/Completeness
    "null.values_found": "נמצאו {count} ערכים ריקים בעמודה '{column}'",
    "null.column_empty": "העמודה '{column}' ריקה לחלוטין",
    "null.above_threshold": "יחס הערכים הריקים ({ratio:.1%}) חורג מהסף ({threshold:.1%}) בעמודה '{column}'",

    # Uniqueness
    "unique.duplicates_found": "נמצאו {count} ערכים כפולים בעמודה '{column}'",
    "unique.composite_duplicates": "נמצאו {count} שילובים כפולים עבור העמודות {columns}",
    "unique.key_violation": "הפרת מפתח ראשי: {count} מפתחות כפולים ב-'{column}'",

    # Type
    "type.mismatch": "אי התאמת סוג בעמודה '{column}': צפוי {expected}, נמצא {actual}",
    "type.coercion_failed": "לא ניתן להמיר {count} ערכים בעמודה '{column}' ל-{target_type}",
    "type.inference_failed": "לא ניתן להסיק את הסוג של העמודה '{column}'",

    # Format
    "format.invalid_email": "נמצאו {count} כתובות דוא\"ל לא תקינות בעמודה '{column}'",
    "format.invalid_phone": "נמצאו {count} מספרי טלפון לא תקינים בעמודה '{column}'",
    "format.invalid_date": "נמצאו {count} תאריכים לא תקינים בעמודה '{column}'",
    "format.invalid_url": "נמצאו {count} כתובות URL לא תקינות בעמודה '{column}'",
    "format.pattern_mismatch": "נמצאו {count} ערכים שאינם תואמים לתבנית '{pattern}' בעמודה '{column}'",
    "format.regex_failed": "אימות ביטוי רגולרי נכשל עבור העמודה '{column}': {count} ערכים לא תואמים",

    # Range
    "range.out_of_bounds": "נמצאו {count} ערכים מחוץ לטווח [{min}, {max}] בעמודה '{column}'",
    "range.below_minimum": "נמצאו {count} ערכים מתחת למינימום ({min}) בעמודה '{column}'",
    "range.above_maximum": "נמצאו {count} ערכים מעל המקסימום ({max}) בעמודה '{column}'",
    "range.outlier_detected": "זוהו {count} חריגים סטטיסטיים בעמודה '{column}'",

    # Referential
    "ref.foreign_key_violation": "נמצאו {count} הפרות מפתח זר בעמודה '{column}'",
    "ref.missing_reference": "נמצאו {count} ערכים ב-'{column}' שאינם קיימים בעמודת ההפניה '{ref_column}'",
    "ref.orphan_records": "נמצאו {count} רשומות יתומות ללא התאמה ב-'{column}'",

    # Statistical
    "stat.distribution_anomaly": "זוהתה חריגה בהתפלגות בעמודה '{column}': {details}",
    "stat.mean_out_of_range": "הערך הממוצע ({mean:.2f}) מחוץ לטווח הצפוי [{min}, {max}] עבור העמודה '{column}'",
    "stat.variance_anomaly": "שונות חריגה ({variance:.2f}) זוהתה בעמודה '{column}'",
    "stat.skewness_anomaly": "העיוות ({skewness:.2f}) מצביע על התפלגות לא נורמלית בעמודה '{column}'",

    # Schema
    "schema.column_missing": "העמודה הצפויה '{column}' חסרה ממערך הנתונים",
    "schema.column_extra": "נמצאה עמודה לא צפויה '{column}' במערך הנתונים",
    "schema.type_mismatch": "לעמודה '{column}' יש סוג {actual}, צפוי {expected}",
    "schema.constraint_violated": "האילוץ '{constraint}' הופר עבור העמודה '{column}'",

    # Cross-table
    "cross.column_mismatch": "הערכים בעמודה '{column1}' אינם תואמים לעמודה '{column2}'",
    "cross.consistency_failed": "בדיקת העקביות בין העמודות נכשלה: {details}",

    # Timeout
    "timeout.exceeded": "האימות פג תוקף לאחר {seconds} שניות עבור '{operation}'",
    "timeout.partial_result": "תוצאות חלקיות הוחזרו עקב פג תוקף: {validated}% מהנתונים אומתו",

    # General
    "validation.failed": "האימות נכשל עבור העמודה '{column}': {reason}",
    "validation.skipped": "האימות דולג עבור העמודה '{column}': {reason}",
    "validation.error": "שגיאה במהלך האימות: {error}",
}


# ==============================================================================
# Persian/Farsi Messages (Complete)
# ==============================================================================

_PERSIAN_MESSAGES: dict[str, str] = {
    # Null/Completeness
    "null.values_found": "{count} مقدار تهی در ستون '{column}' یافت شد",
    "null.column_empty": "ستون '{column}' کاملاً خالی است",
    "null.above_threshold": "نسبت مقادیر تهی ({ratio:.1%}) از آستانه ({threshold:.1%}) در ستون '{column}' فراتر رفته است",

    # Uniqueness
    "unique.duplicates_found": "{count} مقدار تکراری در ستون '{column}' یافت شد",
    "unique.composite_duplicates": "{count} ترکیب تکراری برای ستون‌های {columns} یافت شد",
    "unique.key_violation": "نقض کلید اصلی: {count} کلید تکراری در '{column}'",

    # Type
    "type.mismatch": "عدم تطابق نوع در ستون '{column}': انتظار {expected}، یافت شده {actual}",
    "type.coercion_failed": "امکان تبدیل {count} مقدار در ستون '{column}' به {target_type} وجود ندارد",
    "type.inference_failed": "امکان استنتاج نوع ستون '{column}' وجود ندارد",

    # Format
    "format.invalid_email": "{count} آدرس ایمیل نامعتبر در ستون '{column}' یافت شد",
    "format.invalid_phone": "{count} شماره تلفن نامعتبر در ستون '{column}' یافت شد",
    "format.invalid_date": "{count} تاریخ نامعتبر در ستون '{column}' یافت شد",
    "format.invalid_url": "{count} آدرس URL نامعتبر در ستون '{column}' یافت شد",
    "format.pattern_mismatch": "{count} مقدار که با الگوی '{pattern}' مطابقت ندارند در ستون '{column}' یافت شد",
    "format.regex_failed": "اعتبارسنجی regex برای ستون '{column}' ناموفق بود: {count} مقدار نامطابق",

    # Range
    "range.out_of_bounds": "{count} مقدار خارج از محدوده [{min}, {max}] در ستون '{column}' یافت شد",
    "range.below_minimum": "{count} مقدار کمتر از حداقل ({min}) در ستون '{column}' یافت شد",
    "range.above_maximum": "{count} مقدار بیشتر از حداکثر ({max}) در ستون '{column}' یافت شد",
    "range.outlier_detected": "{count} داده پرت آماری در ستون '{column}' شناسایی شد",

    # General
    "validation.failed": "اعتبارسنجی برای ستون '{column}' ناموفق بود: {reason}",
    "validation.skipped": "اعتبارسنجی برای ستون '{column}' نادیده گرفته شد: {reason}",
    "validation.error": "خطا در حین اعتبارسنجی: {error}",
}


# ==============================================================================
# Catalog Accessor Functions
# ==============================================================================

def get_portuguese_messages() -> ValidatorMessageCatalog:
    """Get Portuguese message catalog.

    Returns:
        Portuguese message catalog
    """
    return ValidatorMessageCatalog.from_dict(
        "pt",
        _PORTUGUESE_MESSAGES,
        metadata={"name": "Português", "complete": True},
    )


def get_portuguese_br_messages() -> ValidatorMessageCatalog:
    """Get Brazilian Portuguese message catalog.

    Returns:
        Brazilian Portuguese message catalog
    """
    messages = {**_PORTUGUESE_MESSAGES, **_PORTUGUESE_BR_MESSAGES}
    return ValidatorMessageCatalog.from_dict(
        "pt-BR",
        messages,
        metadata={"name": "Português Brasileiro", "complete": True, "base": "pt"},
    )


def get_portuguese_pt_messages() -> ValidatorMessageCatalog:
    """Get European Portuguese message catalog.

    Returns:
        European Portuguese message catalog
    """
    messages = {**_PORTUGUESE_MESSAGES, **_PORTUGUESE_PT_MESSAGES}
    return ValidatorMessageCatalog.from_dict(
        "pt-PT",
        messages,
        metadata={"name": "Português Europeu", "complete": True, "base": "pt"},
    )


def get_italian_messages() -> ValidatorMessageCatalog:
    """Get Italian message catalog.

    Returns:
        Italian message catalog
    """
    return ValidatorMessageCatalog.from_dict(
        "it",
        _ITALIAN_MESSAGES,
        metadata={"name": "Italiano", "complete": True},
    )


def get_russian_messages() -> ValidatorMessageCatalog:
    """Get Russian message catalog.

    Returns:
        Russian message catalog
    """
    return ValidatorMessageCatalog.from_dict(
        "ru",
        _RUSSIAN_MESSAGES,
        metadata={"name": "Русский", "complete": True},
    )


def get_arabic_messages() -> ValidatorMessageCatalog:
    """Get Arabic message catalog.

    Returns:
        Arabic message catalog (RTL)
    """
    return ValidatorMessageCatalog.from_dict(
        "ar",
        _ARABIC_MESSAGES,
        metadata={"name": "العربية", "complete": True, "direction": "rtl"},
    )


def get_hebrew_messages() -> ValidatorMessageCatalog:
    """Get Hebrew message catalog.

    Returns:
        Hebrew message catalog (RTL)
    """
    return ValidatorMessageCatalog.from_dict(
        "he",
        _HEBREW_MESSAGES,
        metadata={"name": "עברית", "complete": True, "direction": "rtl"},
    )


def get_persian_messages() -> ValidatorMessageCatalog:
    """Get Persian/Farsi message catalog.

    Returns:
        Persian message catalog (RTL)
    """
    return ValidatorMessageCatalog.from_dict(
        "fa",
        _PERSIAN_MESSAGES,
        metadata={"name": "فارسی", "complete": False, "direction": "rtl"},
    )


def get_all_extended_catalogs() -> dict[str, ValidatorMessageCatalog]:
    """Get all extended catalogs.

    Returns:
        Dictionary of locale code to catalog
    """
    return {
        "pt": get_portuguese_messages(),
        "pt-BR": get_portuguese_br_messages(),
        "pt-PT": get_portuguese_pt_messages(),
        "it": get_italian_messages(),
        "ru": get_russian_messages(),
        "ar": get_arabic_messages(),
        "he": get_hebrew_messages(),
        "fa": get_persian_messages(),
    }


def get_extended_supported_locales() -> list[str]:
    """Get list of extended supported locales.

    Returns:
        List of locale codes
    """
    return ["pt", "pt-BR", "pt-PT", "it", "ru", "ar", "he", "fa"]
