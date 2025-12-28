"""Validator-specific internationalized messages.

This module provides a complete i18n system for validator error messages,
supporting multiple languages and consistent formatting.
"""

from __future__ import annotations

import locale
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidatorMessageCode(str, Enum):
    """Message codes for validator error messages.

    Codes are organized by validator category:
    - NULL: Null/missing value checks
    - UNIQUE: Uniqueness checks
    - TYPE: Data type checks
    - FORMAT: Format/pattern checks
    - RANGE: Value range checks
    - REFERENTIAL: Referential integrity checks
    - STATISTICAL: Statistical checks
    - SCHEMA: Schema validation
    - CUSTOM: Custom validators
    """

    # Null/Completeness checks
    NULL_VALUES_FOUND = "null.values_found"
    NULL_COLUMN_EMPTY = "null.column_empty"
    NULL_ABOVE_THRESHOLD = "null.above_threshold"

    # Uniqueness checks
    UNIQUE_DUPLICATES_FOUND = "unique.duplicates_found"
    UNIQUE_COMPOSITE_DUPLICATES = "unique.composite_duplicates"
    UNIQUE_KEY_VIOLATION = "unique.key_violation"

    # Type checks
    TYPE_MISMATCH = "type.mismatch"
    TYPE_COERCION_FAILED = "type.coercion_failed"
    TYPE_INFERENCE_FAILED = "type.inference_failed"

    # Format/Pattern checks
    FORMAT_INVALID_EMAIL = "format.invalid_email"
    FORMAT_INVALID_PHONE = "format.invalid_phone"
    FORMAT_INVALID_DATE = "format.invalid_date"
    FORMAT_INVALID_URL = "format.invalid_url"
    FORMAT_PATTERN_MISMATCH = "format.pattern_mismatch"
    FORMAT_REGEX_FAILED = "format.regex_failed"

    # Range checks
    RANGE_OUT_OF_BOUNDS = "range.out_of_bounds"
    RANGE_BELOW_MINIMUM = "range.below_minimum"
    RANGE_ABOVE_MAXIMUM = "range.above_maximum"
    RANGE_OUTLIER_DETECTED = "range.outlier_detected"

    # Referential integrity
    REF_FOREIGN_KEY_VIOLATION = "ref.foreign_key_violation"
    REF_MISSING_REFERENCE = "ref.missing_reference"
    REF_ORPHAN_RECORDS = "ref.orphan_records"

    # Statistical checks
    STAT_DISTRIBUTION_ANOMALY = "stat.distribution_anomaly"
    STAT_MEAN_OUT_OF_RANGE = "stat.mean_out_of_range"
    STAT_VARIANCE_ANOMALY = "stat.variance_anomaly"
    STAT_SKEWNESS_ANOMALY = "stat.skewness_anomaly"

    # Schema validation
    SCHEMA_COLUMN_MISSING = "schema.column_missing"
    SCHEMA_COLUMN_EXTRA = "schema.column_extra"
    SCHEMA_TYPE_MISMATCH = "schema.type_mismatch"
    SCHEMA_CONSTRAINT_VIOLATED = "schema.constraint_violated"

    # Cross-table checks
    CROSS_COLUMN_MISMATCH = "cross.column_mismatch"
    CROSS_CONSISTENCY_FAILED = "cross.consistency_failed"

    # Timeout/Performance
    TIMEOUT_EXCEEDED = "timeout.exceeded"
    TIMEOUT_PARTIAL_RESULT = "timeout.partial_result"

    # General
    VALIDATION_FAILED = "validation.failed"
    VALIDATION_SKIPPED = "validation.skipped"
    VALIDATION_ERROR = "validation.error"


# Default English messages
_DEFAULT_MESSAGES: dict[str, str] = {
    # Null/Completeness
    "null.values_found": "Found {count} null values in column '{column}'",
    "null.column_empty": "Column '{column}' is completely empty",
    "null.above_threshold": "Null ratio ({ratio:.1%}) exceeds threshold ({threshold:.1%}) in column '{column}'",

    # Uniqueness
    "unique.duplicates_found": "Found {count} duplicate values in column '{column}'",
    "unique.composite_duplicates": "Found {count} duplicate combinations for columns {columns}",
    "unique.key_violation": "Primary key violation: {count} duplicate keys in '{column}'",

    # Type
    "type.mismatch": "Type mismatch in column '{column}': expected {expected}, found {actual}",
    "type.coercion_failed": "Could not convert {count} values in column '{column}' to {target_type}",
    "type.inference_failed": "Could not infer type for column '{column}'",

    # Format
    "format.invalid_email": "Found {count} invalid email addresses in column '{column}'",
    "format.invalid_phone": "Found {count} invalid phone numbers in column '{column}'",
    "format.invalid_date": "Found {count} invalid dates in column '{column}'",
    "format.invalid_url": "Found {count} invalid URLs in column '{column}'",
    "format.pattern_mismatch": "Found {count} values not matching pattern '{pattern}' in column '{column}'",
    "format.regex_failed": "Regex validation failed for column '{column}': {count} non-matching values",

    # Range
    "range.out_of_bounds": "Found {count} values outside range [{min}, {max}] in column '{column}'",
    "range.below_minimum": "Found {count} values below minimum ({min}) in column '{column}'",
    "range.above_maximum": "Found {count} values above maximum ({max}) in column '{column}'",
    "range.outlier_detected": "Detected {count} statistical outliers in column '{column}'",

    # Referential
    "ref.foreign_key_violation": "Found {count} foreign key violations in column '{column}'",
    "ref.missing_reference": "Found {count} values in '{column}' not present in reference column '{ref_column}'",
    "ref.orphan_records": "Found {count} orphan records without matching parent in '{column}'",

    # Statistical
    "stat.distribution_anomaly": "Distribution anomaly detected in column '{column}': {details}",
    "stat.mean_out_of_range": "Mean value ({mean:.2f}) is outside expected range [{min}, {max}] for column '{column}'",
    "stat.variance_anomaly": "Unusual variance ({variance:.2f}) detected in column '{column}'",
    "stat.skewness_anomaly": "Skewness ({skewness:.2f}) indicates non-normal distribution in column '{column}'",

    # Schema
    "schema.column_missing": "Expected column '{column}' is missing from the dataset",
    "schema.column_extra": "Unexpected column '{column}' found in dataset",
    "schema.type_mismatch": "Column '{column}' has type {actual}, expected {expected}",
    "schema.constraint_violated": "Constraint '{constraint}' violated for column '{column}'",

    # Cross-table
    "cross.column_mismatch": "Values in column '{column1}' do not match column '{column2}'",
    "cross.consistency_failed": "Cross-column consistency check failed: {details}",

    # Timeout
    "timeout.exceeded": "Validation timed out after {seconds}s for '{operation}'",
    "timeout.partial_result": "Partial results returned due to timeout: validated {validated}% of data",

    # General
    "validation.failed": "Validation failed for column '{column}': {reason}",
    "validation.skipped": "Validation skipped for column '{column}': {reason}",
    "validation.error": "Error during validation: {error}",
}

# Korean messages
_KOREAN_MESSAGES: dict[str, str] = {
    # Null/완전성
    "null.values_found": "'{column}' 컬럼에서 {count}개의 null 값이 발견되었습니다",
    "null.column_empty": "'{column}' 컬럼이 완전히 비어있습니다",
    "null.above_threshold": "'{column}' 컬럼의 null 비율({ratio:.1%})이 임계값({threshold:.1%})을 초과했습니다",

    # 고유성
    "unique.duplicates_found": "'{column}' 컬럼에서 {count}개의 중복 값이 발견되었습니다",
    "unique.composite_duplicates": "{columns} 컬럼에서 {count}개의 중복 조합이 발견되었습니다",
    "unique.key_violation": "기본키 위반: '{column}'에서 {count}개의 중복 키가 발견되었습니다",

    # 타입
    "type.mismatch": "'{column}' 컬럼의 타입 불일치: {expected} 예상, {actual} 발견",
    "type.coercion_failed": "'{column}' 컬럼에서 {count}개의 값을 {target_type}으로 변환할 수 없습니다",
    "type.inference_failed": "'{column}' 컬럼의 타입을 추론할 수 없습니다",

    # 형식
    "format.invalid_email": "'{column}' 컬럼에서 {count}개의 잘못된 이메일 주소가 발견되었습니다",
    "format.invalid_phone": "'{column}' 컬럼에서 {count}개의 잘못된 전화번호가 발견되었습니다",
    "format.invalid_date": "'{column}' 컬럼에서 {count}개의 잘못된 날짜가 발견되었습니다",
    "format.invalid_url": "'{column}' 컬럼에서 {count}개의 잘못된 URL이 발견되었습니다",
    "format.pattern_mismatch": "'{column}' 컬럼에서 패턴 '{pattern}'과 일치하지 않는 {count}개의 값이 발견되었습니다",
    "format.regex_failed": "'{column}' 컬럼의 정규식 검증 실패: {count}개의 불일치 값",

    # 범위
    "range.out_of_bounds": "'{column}' 컬럼에서 범위 [{min}, {max}]를 벗어난 {count}개의 값이 발견되었습니다",
    "range.below_minimum": "'{column}' 컬럼에서 최소값({min}) 미만인 {count}개의 값이 발견되었습니다",
    "range.above_maximum": "'{column}' 컬럼에서 최대값({max}) 초과인 {count}개의 값이 발견되었습니다",
    "range.outlier_detected": "'{column}' 컬럼에서 {count}개의 통계적 이상치가 감지되었습니다",

    # 참조 무결성
    "ref.foreign_key_violation": "'{column}' 컬럼에서 {count}개의 외래키 위반이 발견되었습니다",
    "ref.missing_reference": "'{column}'의 {count}개 값이 참조 컬럼 '{ref_column}'에 존재하지 않습니다",
    "ref.orphan_records": "'{column}'에서 상위 레코드와 일치하지 않는 {count}개의 고아 레코드가 발견되었습니다",

    # 통계
    "stat.distribution_anomaly": "'{column}' 컬럼에서 분포 이상이 감지되었습니다: {details}",
    "stat.mean_out_of_range": "'{column}' 컬럼의 평균값({mean:.2f})이 예상 범위 [{min}, {max}]를 벗어났습니다",
    "stat.variance_anomaly": "'{column}' 컬럼에서 비정상적인 분산({variance:.2f})이 감지되었습니다",
    "stat.skewness_anomaly": "'{column}' 컬럼의 왜도({skewness:.2f})가 비정규 분포를 나타냅니다",

    # 스키마
    "schema.column_missing": "예상 컬럼 '{column}'이(가) 데이터셋에서 누락되었습니다",
    "schema.column_extra": "예상치 못한 컬럼 '{column}'이(가) 데이터셋에서 발견되었습니다",
    "schema.type_mismatch": "'{column}' 컬럼의 타입이 {actual}이지만, {expected}이어야 합니다",
    "schema.constraint_violated": "'{column}' 컬럼에서 제약조건 '{constraint}'이(가) 위반되었습니다",

    # 교차 테이블
    "cross.column_mismatch": "'{column1}' 컬럼의 값이 '{column2}' 컬럼과 일치하지 않습니다",
    "cross.consistency_failed": "교차 컬럼 일관성 검사 실패: {details}",

    # 타임아웃
    "timeout.exceeded": "'{operation}'에 대한 검증이 {seconds}초 후 타임아웃되었습니다",
    "timeout.partial_result": "타임아웃으로 인해 부분 결과 반환: 데이터의 {validated}% 검증됨",

    # 일반
    "validation.failed": "'{column}' 컬럼 검증 실패: {reason}",
    "validation.skipped": "'{column}' 컬럼 검증 건너뜀: {reason}",
    "validation.error": "검증 중 오류 발생: {error}",
}

# Japanese messages
_JAPANESE_MESSAGES: dict[str, str] = {
    "null.values_found": "列'{column}'で{count}個のnull値が見つかりました",
    "null.column_empty": "列'{column}'は完全に空です",
    "null.above_threshold": "列'{column}'のnull比率({ratio:.1%})が閾値({threshold:.1%})を超えています",

    "unique.duplicates_found": "列'{column}'で{count}個の重複値が見つかりました",
    "unique.composite_duplicates": "列{columns}で{count}個の重複組み合わせが見つかりました",
    "unique.key_violation": "主キー違反: '{column}'で{count}個の重複キーが見つかりました",

    "type.mismatch": "列'{column}'の型不一致: {expected}を期待、{actual}が見つかりました",
    "type.coercion_failed": "列'{column}'の{count}個の値を{target_type}に変換できません",
    "type.inference_failed": "列'{column}'の型を推論できません",

    "format.invalid_email": "列'{column}'で{count}個の無効なメールアドレスが見つかりました",
    "format.invalid_phone": "列'{column}'で{count}個の無効な電話番号が見つかりました",
    "format.invalid_date": "列'{column}'で{count}個の無効な日付が見つかりました",
    "format.invalid_url": "列'{column}'で{count}個の無効なURLが見つかりました",
    "format.pattern_mismatch": "列'{column}'でパターン'{pattern}'に一致しない{count}個の値が見つかりました",
    "format.regex_failed": "列'{column}'の正規表現検証失敗: {count}個の不一致値",

    "range.out_of_bounds": "列'{column}'で範囲[{min}, {max}]外の{count}個の値が見つかりました",
    "range.below_minimum": "列'{column}'で最小値({min})未満の{count}個の値が見つかりました",
    "range.above_maximum": "列'{column}'で最大値({max})超過の{count}個の値が見つかりました",
    "range.outlier_detected": "列'{column}'で{count}個の統計的外れ値が検出されました",

    "ref.foreign_key_violation": "列'{column}'で{count}個の外部キー違反が見つかりました",
    "ref.missing_reference": "'{column}'の{count}個の値が参照列'{ref_column}'に存在しません",
    "ref.orphan_records": "'{column}'で親レコードと一致しない{count}個の孤立レコードが見つかりました",

    "stat.distribution_anomaly": "列'{column}'で分布異常が検出されました: {details}",
    "stat.mean_out_of_range": "列'{column}'の平均値({mean:.2f})が予想範囲[{min}, {max}]外です",
    "stat.variance_anomaly": "列'{column}'で異常な分散({variance:.2f})が検出されました",
    "stat.skewness_anomaly": "列'{column}'の歪度({skewness:.2f})は非正規分布を示しています",

    "schema.column_missing": "期待される列'{column}'がデータセットにありません",
    "schema.column_extra": "予期しない列'{column}'がデータセットに見つかりました",
    "schema.type_mismatch": "列'{column}'の型は{actual}ですが、{expected}であるべきです",
    "schema.constraint_violated": "列'{column}'で制約'{constraint}'が違反されました",

    "cross.column_mismatch": "列'{column1}'の値が列'{column2}'と一致しません",
    "cross.consistency_failed": "クロス列一貫性チェック失敗: {details}",

    "timeout.exceeded": "'{operation}'の検証が{seconds}秒後にタイムアウトしました",
    "timeout.partial_result": "タイムアウトにより部分結果を返却: データの{validated}%を検証",

    "validation.failed": "列'{column}'の検証失敗: {reason}",
    "validation.skipped": "列'{column}'の検証スキップ: {reason}",
    "validation.error": "検証中にエラーが発生しました: {error}",
}

# Chinese (Simplified) messages
_CHINESE_MESSAGES: dict[str, str] = {
    "null.values_found": "在列'{column}'中发现了{count}个空值",
    "null.column_empty": "列'{column}'完全为空",
    "null.above_threshold": "列'{column}'的空值比例({ratio:.1%})超过了阈值({threshold:.1%})",

    "unique.duplicates_found": "在列'{column}'中发现了{count}个重复值",
    "unique.composite_duplicates": "在列{columns}中发现了{count}个重复组合",
    "unique.key_violation": "主键违规: 在'{column}'中发现了{count}个重复键",

    "type.mismatch": "列'{column}'类型不匹配: 期望{expected}，发现{actual}",
    "type.coercion_failed": "无法将列'{column}'中的{count}个值转换为{target_type}",
    "type.inference_failed": "无法推断列'{column}'的类型",

    "format.invalid_email": "在列'{column}'中发现了{count}个无效的电子邮件地址",
    "format.invalid_phone": "在列'{column}'中发现了{count}个无效的电话号码",
    "format.invalid_date": "在列'{column}'中发现了{count}个无效的日期",
    "format.invalid_url": "在列'{column}'中发现了{count}个无效的URL",
    "format.pattern_mismatch": "在列'{column}'中发现了{count}个不匹配模式'{pattern}'的值",
    "format.regex_failed": "列'{column}'的正则表达式验证失败: {count}个不匹配的值",

    "range.out_of_bounds": "在列'{column}'中发现了{count}个超出范围[{min}, {max}]的值",
    "range.below_minimum": "在列'{column}'中发现了{count}个低于最小值({min})的值",
    "range.above_maximum": "在列'{column}'中发现了{count}个超过最大值({max})的值",
    "range.outlier_detected": "在列'{column}'中检测到{count}个统计异常值",

    "validation.failed": "列'{column}'验证失败: {reason}",
    "validation.skipped": "跳过列'{column}'的验证: {reason}",
    "validation.error": "验证过程中发生错误: {error}",
}

# German messages
_GERMAN_MESSAGES: dict[str, str] = {
    "null.values_found": "{count} Nullwerte in Spalte '{column}' gefunden",
    "null.column_empty": "Spalte '{column}' ist vollständig leer",
    "null.above_threshold": "Nullanteil ({ratio:.1%}) überschreitet Schwellenwert ({threshold:.1%}) in Spalte '{column}'",

    "unique.duplicates_found": "{count} Duplikate in Spalte '{column}' gefunden",
    "unique.key_violation": "Primärschlüsselverletzung: {count} doppelte Schlüssel in '{column}'",

    "type.mismatch": "Typabweichung in Spalte '{column}': erwartet {expected}, gefunden {actual}",
    "format.pattern_mismatch": "{count} Werte in Spalte '{column}' entsprechen nicht dem Muster '{pattern}'",

    "validation.failed": "Validierung für Spalte '{column}' fehlgeschlagen: {reason}",
    "validation.error": "Fehler während der Validierung: {error}",
}

# French messages
_FRENCH_MESSAGES: dict[str, str] = {
    "null.values_found": "{count} valeurs nulles trouvées dans la colonne '{column}'",
    "null.column_empty": "La colonne '{column}' est entièrement vide",

    "unique.duplicates_found": "{count} valeurs en double trouvées dans la colonne '{column}'",
    "type.mismatch": "Incompatibilité de type dans la colonne '{column}': attendu {expected}, trouvé {actual}",

    "validation.failed": "Échec de la validation pour la colonne '{column}': {reason}",
    "validation.error": "Erreur lors de la validation: {error}",
}

# Spanish messages
_SPANISH_MESSAGES: dict[str, str] = {
    "null.values_found": "Se encontraron {count} valores nulos en la columna '{column}'",
    "null.column_empty": "La columna '{column}' está completamente vacía",

    "unique.duplicates_found": "Se encontraron {count} valores duplicados en la columna '{column}'",
    "type.mismatch": "Discrepancia de tipo en la columna '{column}': esperado {expected}, encontrado {actual}",

    "validation.failed": "Falló la validación para la columna '{column}': {reason}",
    "validation.error": "Error durante la validación: {error}",
}


@dataclass
class ValidatorI18n:
    """Internationalization manager for validator messages.

    Thread-safe singleton that manages locale and message resolution.

    Example:
        i18n = ValidatorI18n.get_instance()
        i18n.set_locale("ko")
        msg = i18n.get_message(ValidatorMessageCode.NULL_VALUES_FOUND, column="email", count=5)
    """

    _locale: str = "en"
    _fallback_locale: str = "en"
    _catalogs: dict[str, dict[str, str]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    _instance: "ValidatorI18n | None" = None

    @classmethod
    def get_instance(cls) -> "ValidatorI18n":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._initialize_catalogs()
        return cls._instance

    def _initialize_catalogs(self) -> None:
        """Initialize message catalogs."""
        self._catalogs = {
            "en": _DEFAULT_MESSAGES,
            "ko": _KOREAN_MESSAGES,
            "ja": _JAPANESE_MESSAGES,
            "zh": _CHINESE_MESSAGES,
            "de": _GERMAN_MESSAGES,
            "fr": _FRENCH_MESSAGES,
            "es": _SPANISH_MESSAGES,
        }

    def set_locale(self, locale_code: str) -> None:
        """Set the current locale.

        Args:
            locale_code: Locale code (e.g., "en", "ko", "ja")
        """
        with self._lock:
            # Extract language code (e.g., "ko_KR" -> "ko")
            lang = locale_code.split("_")[0].lower()
            if lang in self._catalogs:
                self._locale = lang
            else:
                self._locale = self._fallback_locale

    def get_locale(self) -> str:
        """Get the current locale.

        Returns:
            Current locale code
        """
        return self._locale

    def get_message(
        self,
        code: ValidatorMessageCode | str,
        **kwargs: Any,
    ) -> str:
        """Get a localized message.

        Args:
            code: Message code
            **kwargs: Format arguments

        Returns:
            Formatted localized message
        """
        key = code.value if isinstance(code, ValidatorMessageCode) else code

        # Try current locale
        catalog = self._catalogs.get(self._locale, {})
        template = catalog.get(key)

        # Fallback to English
        if template is None:
            catalog = self._catalogs.get(self._fallback_locale, {})
            template = catalog.get(key, f"[{key}]")

        # Format message
        try:
            return template.format(**kwargs)
        except KeyError as e:
            # Missing placeholder
            return f"{template} (missing: {e})"
        except Exception:
            return template

    def add_catalog(self, locale_code: str, messages: dict[str, str]) -> None:
        """Add or update a message catalog.

        Args:
            locale_code: Locale code
            messages: Dictionary of message key -> template
        """
        with self._lock:
            lang = locale_code.split("_")[0].lower()
            if lang in self._catalogs:
                self._catalogs[lang].update(messages)
            else:
                self._catalogs[lang] = messages


# Module-level convenience functions

def get_validator_message(
    code: ValidatorMessageCode | str,
    **kwargs: Any,
) -> str:
    """Get a localized validator message.

    Args:
        code: Message code
        **kwargs: Format arguments

    Returns:
        Formatted message
    """
    return ValidatorI18n.get_instance().get_message(code, **kwargs)


def set_validator_locale(locale_code: str) -> None:
    """Set the validator locale.

    Args:
        locale_code: Locale code (e.g., "en", "ko", "ja")
    """
    ValidatorI18n.get_instance().set_locale(locale_code)


def get_validator_locale() -> str:
    """Get the current validator locale.

    Returns:
        Current locale code
    """
    return ValidatorI18n.get_instance().get_locale()


def format_issue_message(
    issue_type: str,
    column: str,
    count: int,
    **extra: Any,
) -> str:
    """Format a validation issue message.

    This function maps common issue types to their message codes
    and formats them with the provided arguments.

    Args:
        issue_type: Issue type (e.g., "null", "duplicate", "format")
        column: Column name
        count: Number of issues
        **extra: Additional format arguments

    Returns:
        Formatted message
    """
    # Map issue types to message codes
    type_to_code = {
        "null": ValidatorMessageCode.NULL_VALUES_FOUND,
        "null_value": ValidatorMessageCode.NULL_VALUES_FOUND,
        "duplicate": ValidatorMessageCode.UNIQUE_DUPLICATES_FOUND,
        "type_mismatch": ValidatorMessageCode.TYPE_MISMATCH,
        "invalid_email": ValidatorMessageCode.FORMAT_INVALID_EMAIL,
        "invalid_phone": ValidatorMessageCode.FORMAT_INVALID_PHONE,
        "invalid_date": ValidatorMessageCode.FORMAT_INVALID_DATE,
        "pattern_mismatch": ValidatorMessageCode.FORMAT_PATTERN_MISMATCH,
        "out_of_range": ValidatorMessageCode.RANGE_OUT_OF_BOUNDS,
        "outlier": ValidatorMessageCode.RANGE_OUTLIER_DETECTED,
        "foreign_key_violation": ValidatorMessageCode.REF_FOREIGN_KEY_VIOLATION,
        "schema_mismatch": ValidatorMessageCode.SCHEMA_TYPE_MISMATCH,
    }

    code = type_to_code.get(issue_type.lower(), ValidatorMessageCode.VALIDATION_FAILED)
    return get_validator_message(code, column=column, count=count, **extra)
