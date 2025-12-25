"""Custom pattern configuration via YAML files.

This module provides a flexible system for defining custom validation patterns
through YAML configuration files, eliminating the need for code changes.

Key features:
- YAML-based pattern definitions
- Hierarchical pattern organization
- Pattern inheritance and composition
- Hot-reload support for development
- Pattern validation and testing

Example YAML configuration:
    patterns:
      korean_phone:
        name: Korean Phone Number
        regex: "^01[0-9]-[0-9]{3,4}-[0-9]{4}$"
        priority: 90
        data_type: korean_phone
        examples:
          - "010-1234-5678"
          - "011-123-4567"

      email:
        name: Email Address
        regex: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
        priority: 85
        data_type: email

Example usage:
    from truthound.profiler.custom_patterns import (
        PatternConfig,
        load_patterns,
        PatternConfigLoader,
    )

    # Load patterns from YAML
    patterns = load_patterns("patterns.yaml")

    # Or use the loader for more control
    loader = PatternConfigLoader()
    loader.load_file("patterns.yaml")
    loader.load_directory("patterns/")

    # Get all patterns
    all_patterns = loader.get_all_patterns()
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from truthound.profiler.base import DataType


# =============================================================================
# Pattern Configuration Types
# =============================================================================


class PatternPriority(int, Enum):
    """Priority levels for pattern matching."""

    HIGHEST = 100
    HIGH = 90
    MEDIUM = 50
    LOW = 25
    LOWEST = 10


@dataclass
class PatternExample:
    """Example value for pattern testing."""

    value: str
    should_match: bool = True
    description: str = ""


@dataclass
class PatternConfig:
    """Configuration for a single pattern.

    Attributes:
        name: Human-readable pattern name
        pattern_id: Unique identifier for the pattern
        regex: Regular expression pattern
        priority: Matching priority (higher = checked first)
        data_type: Inferred data type when pattern matches
        min_match_ratio: Minimum ratio of values that must match
        description: Pattern description
        examples: Example values for testing
        tags: Tags for categorization
        enabled: Whether pattern is active
        case_sensitive: Whether regex is case-sensitive
        multiline: Whether regex uses multiline mode
        validator_fn: Optional custom validator function name
        metadata: Additional metadata
    """

    name: str
    pattern_id: str
    regex: str
    priority: int = PatternPriority.MEDIUM
    data_type: str = "string"
    min_match_ratio: float = 0.8
    description: str = ""
    examples: list[PatternExample] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    enabled: bool = True
    case_sensitive: bool = True
    multiline: bool = False
    validator_fn: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Internal
    _compiled_regex: re.Pattern | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Compile regex pattern."""
        self._compile_regex()

    def _compile_regex(self) -> None:
        """Compile the regex pattern."""
        flags = 0
        if not self.case_sensitive:
            flags |= re.IGNORECASE
        if self.multiline:
            flags |= re.MULTILINE

        try:
            self._compiled_regex = re.compile(self.regex, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex for pattern '{self.name}': {e}")

    @property
    def compiled_regex(self) -> re.Pattern:
        """Get compiled regex pattern."""
        if self._compiled_regex is None:
            self._compile_regex()
        return self._compiled_regex  # type: ignore

    def matches(self, value: str) -> bool:
        """Check if value matches the pattern."""
        if value is None:
            return False
        try:
            return bool(self.compiled_regex.match(str(value)))
        except Exception:
            return False

    def get_data_type(self) -> DataType:
        """Get the DataType enum value."""
        try:
            return DataType(self.data_type)
        except ValueError:
            return DataType.STRING

    def validate_examples(self) -> list[tuple[str, bool, str]]:
        """Validate all examples against the pattern.

        Returns:
            List of (value, passed, message) tuples
        """
        results = []
        for example in self.examples:
            actual = self.matches(example.value)
            passed = actual == example.should_match

            if passed:
                message = "OK"
            else:
                expected = "match" if example.should_match else "not match"
                got = "matched" if actual else "did not match"
                message = f"Expected {expected}, but {got}"

            results.append((example.value, passed, message))

        return results

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "pattern_id": self.pattern_id,
            "regex": self.regex,
            "priority": self.priority,
            "data_type": self.data_type,
            "min_match_ratio": self.min_match_ratio,
            "description": self.description,
            "examples": [
                {
                    "value": e.value,
                    "should_match": e.should_match,
                    "description": e.description,
                }
                for e in self.examples
            ],
            "tags": self.tags,
            "enabled": self.enabled,
            "case_sensitive": self.case_sensitive,
            "multiline": self.multiline,
            "validator_fn": self.validator_fn,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], pattern_id: str | None = None) -> "PatternConfig":
        """Create from dictionary."""
        examples = []
        for ex in data.get("examples", []):
            if isinstance(ex, str):
                examples.append(PatternExample(value=ex))
            elif isinstance(ex, dict):
                examples.append(PatternExample(
                    value=ex.get("value", ""),
                    should_match=ex.get("should_match", True),
                    description=ex.get("description", ""),
                ))

        return cls(
            name=data.get("name", pattern_id or "unnamed"),
            pattern_id=pattern_id or data.get("pattern_id", data.get("name", "unnamed")),
            regex=data["regex"],
            priority=data.get("priority", PatternPriority.MEDIUM),
            data_type=data.get("data_type", "string"),
            min_match_ratio=data.get("min_match_ratio", 0.8),
            description=data.get("description", ""),
            examples=examples,
            tags=data.get("tags", []),
            enabled=data.get("enabled", True),
            case_sensitive=data.get("case_sensitive", True),
            multiline=data.get("multiline", False),
            validator_fn=data.get("validator_fn"),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Pattern Group Configuration
# =============================================================================


@dataclass
class PatternGroup:
    """Group of related patterns.

    Allows organizing patterns into logical categories.
    """

    name: str
    group_id: str
    description: str = ""
    patterns: list[PatternConfig] = field(default_factory=list)
    enabled: bool = True
    priority_boost: int = 0  # Added to all pattern priorities
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_patterns(self, include_disabled: bool = False) -> list[PatternConfig]:
        """Get patterns in this group.

        Args:
            include_disabled: Whether to include disabled patterns

        Returns:
            List of patterns
        """
        if not self.enabled:
            return []

        patterns = []
        for p in self.patterns:
            if p.enabled or include_disabled:
                # Apply priority boost
                if self.priority_boost != 0:
                    p = PatternConfig(
                        **{
                            **p.to_dict(),
                            "priority": p.priority + self.priority_boost,
                        }
                    )
                patterns.append(p)

        return patterns

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "group_id": self.group_id,
            "description": self.description,
            "patterns": [p.to_dict() for p in self.patterns],
            "enabled": self.enabled,
            "priority_boost": self.priority_boost,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], group_id: str | None = None) -> "PatternGroup":
        """Create from dictionary."""
        patterns = []
        for pattern_id, pattern_data in data.get("patterns", {}).items():
            patterns.append(PatternConfig.from_dict(pattern_data, pattern_id))

        return cls(
            name=data.get("name", group_id or "unnamed"),
            group_id=group_id or data.get("group_id", data.get("name", "unnamed")),
            description=data.get("description", ""),
            patterns=patterns,
            enabled=data.get("enabled", True),
            priority_boost=data.get("priority_boost", 0),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# YAML Configuration Schema
# =============================================================================


@dataclass
class PatternConfigSchema:
    """Complete pattern configuration schema.

    Represents a full YAML configuration file.
    """

    version: str = "1.0"
    name: str = ""
    description: str = ""
    patterns: dict[str, PatternConfig] = field(default_factory=dict)
    groups: dict[str, PatternGroup] = field(default_factory=dict)
    extends: list[str] = field(default_factory=list)  # Parent configs to inherit from
    metadata: dict[str, Any] = field(default_factory=dict)
    loaded_at: datetime = field(default_factory=datetime.now)
    source_path: str = ""

    def get_all_patterns(self, include_disabled: bool = False) -> list[PatternConfig]:
        """Get all patterns from this configuration.

        Args:
            include_disabled: Whether to include disabled patterns

        Returns:
            List of all patterns, sorted by priority
        """
        all_patterns = []

        # Add standalone patterns
        for p in self.patterns.values():
            if p.enabled or include_disabled:
                all_patterns.append(p)

        # Add patterns from groups
        for group in self.groups.values():
            all_patterns.extend(group.get_patterns(include_disabled))

        # Sort by priority (highest first)
        return sorted(all_patterns, key=lambda p: p.priority, reverse=True)

    def get_pattern(self, pattern_id: str) -> PatternConfig | None:
        """Get a specific pattern by ID."""
        if pattern_id in self.patterns:
            return self.patterns[pattern_id]

        for group in self.groups.values():
            for p in group.patterns:
                if p.pattern_id == pattern_id:
                    return p

        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "patterns": {pid: p.to_dict() for pid, p in self.patterns.items()},
            "groups": {gid: g.to_dict() for gid, g in self.groups.items()},
            "extends": self.extends,
            "metadata": self.metadata,
        }


# =============================================================================
# YAML Parser
# =============================================================================


def _parse_yaml(content: str) -> dict[str, Any]:
    """Parse YAML content.

    Supports PyYAML if available, otherwise uses basic parsing.
    """
    try:
        import yaml
        return yaml.safe_load(content) or {}
    except ImportError:
        # Basic YAML parsing for simple structures
        return _basic_yaml_parse(content)


def _basic_yaml_parse(content: str) -> dict[str, Any]:
    """Basic YAML parser for simple structures.

    This is a fallback when PyYAML is not installed.
    Only supports simple key-value pairs and lists.
    """
    import json

    # Try JSON first (YAML is a superset)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Very basic YAML parsing
    result: dict[str, Any] = {}
    current_key: str | None = None
    current_indent = 0

    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip())

        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()

            if value:
                # Remove quotes
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                result[key] = value
            else:
                result[key] = {}
                current_key = key
                current_indent = indent

    return result


def _dump_yaml(data: dict[str, Any]) -> str:
    """Dump dictionary to YAML string."""
    try:
        import yaml
        return yaml.dump(data, default_flow_style=False, allow_unicode=True)
    except ImportError:
        import json
        return json.dumps(data, indent=2, ensure_ascii=False)


# =============================================================================
# Configuration Loader
# =============================================================================


class PatternConfigLoader:
    """Loads pattern configurations from YAML files.

    Supports:
    - Single file loading
    - Directory scanning
    - Configuration inheritance
    - Hot-reload for development

    Example:
        loader = PatternConfigLoader()
        loader.load_file("patterns.yaml")
        loader.load_directory("patterns/")

        # Get all patterns
        patterns = loader.get_all_patterns()

        # Enable hot-reload
        loader.enable_hot_reload(interval=5.0)
    """

    def __init__(
        self,
        auto_validate: bool = True,
        strict_mode: bool = False,
    ):
        """Initialize loader.

        Args:
            auto_validate: Validate patterns on load
            strict_mode: Fail on any validation error
        """
        self.auto_validate = auto_validate
        self.strict_mode = strict_mode

        self._configs: dict[str, PatternConfigSchema] = {}
        self._file_mtimes: dict[str, float] = {}
        self._hot_reload_enabled = False
        self._hot_reload_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stop_hot_reload = threading.Event()

    def load_file(self, path: str | Path) -> PatternConfigSchema:
        """Load patterns from a YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Loaded configuration
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pattern config not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        data = _parse_yaml(content)
        config = self._parse_config(data, str(path))

        if self.auto_validate:
            errors = self.validate_config(config)
            if errors and self.strict_mode:
                raise ValueError(f"Pattern validation errors: {errors}")

        with self._lock:
            self._configs[str(path)] = config
            self._file_mtimes[str(path)] = path.stat().st_mtime

        return config

    def load_directory(
        self,
        directory: str | Path,
        pattern: str = "*.yaml",
        recursive: bool = True,
    ) -> list[PatternConfigSchema]:
        """Load all pattern files from a directory.

        Args:
            directory: Directory to scan
            pattern: Glob pattern for files
            recursive: Whether to search recursively

        Returns:
            List of loaded configurations
        """
        directory = Path(directory)
        if not directory.exists():
            return []

        configs = []
        glob_method = directory.rglob if recursive else directory.glob

        for file_path in glob_method(pattern):
            if file_path.is_file():
                try:
                    config = self.load_file(file_path)
                    configs.append(config)
                except Exception as e:
                    if self.strict_mode:
                        raise
                    # Log warning and continue

        # Also try .yml extension
        if pattern.endswith(".yaml"):
            yml_pattern = pattern.replace(".yaml", ".yml")
            for file_path in glob_method(yml_pattern):
                if file_path.is_file() and str(file_path) not in self._configs:
                    try:
                        config = self.load_file(file_path)
                        configs.append(config)
                    except Exception as e:
                        if self.strict_mode:
                            raise

        return configs

    def load_from_string(self, content: str, name: str = "inline") -> PatternConfigSchema:
        """Load patterns from a YAML string.

        Args:
            content: YAML content
            name: Name for the configuration

        Returns:
            Loaded configuration
        """
        data = _parse_yaml(content)
        config = self._parse_config(data, name)

        with self._lock:
            self._configs[name] = config

        return config

    def _parse_config(self, data: dict[str, Any], source: str) -> PatternConfigSchema:
        """Parse configuration data into schema object."""
        # Parse standalone patterns
        patterns = {}
        for pattern_id, pattern_data in data.get("patterns", {}).items():
            if isinstance(pattern_data, dict):
                patterns[pattern_id] = PatternConfig.from_dict(pattern_data, pattern_id)

        # Parse groups
        groups = {}
        for group_id, group_data in data.get("groups", {}).items():
            if isinstance(group_data, dict):
                groups[group_id] = PatternGroup.from_dict(group_data, group_id)

        return PatternConfigSchema(
            version=data.get("version", "1.0"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            patterns=patterns,
            groups=groups,
            extends=data.get("extends", []),
            metadata=data.get("metadata", {}),
            source_path=source,
        )

    def get_all_patterns(self, include_disabled: bool = False) -> list[PatternConfig]:
        """Get all loaded patterns.

        Args:
            include_disabled: Whether to include disabled patterns

        Returns:
            List of all patterns, sorted by priority
        """
        all_patterns = []

        with self._lock:
            for config in self._configs.values():
                all_patterns.extend(config.get_all_patterns(include_disabled))

        # Remove duplicates by pattern_id (keep highest priority)
        seen: dict[str, PatternConfig] = {}
        for p in all_patterns:
            if p.pattern_id not in seen or p.priority > seen[p.pattern_id].priority:
                seen[p.pattern_id] = p

        return sorted(seen.values(), key=lambda p: p.priority, reverse=True)

    def get_pattern(self, pattern_id: str) -> PatternConfig | None:
        """Get a specific pattern by ID.

        Args:
            pattern_id: Pattern identifier

        Returns:
            Pattern config or None
        """
        with self._lock:
            for config in self._configs.values():
                pattern = config.get_pattern(pattern_id)
                if pattern:
                    return pattern
        return None

    def get_patterns_by_tag(self, tag: str) -> list[PatternConfig]:
        """Get patterns with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of matching patterns
        """
        return [p for p in self.get_all_patterns() if tag in p.tags]

    def get_patterns_by_type(self, data_type: str | DataType) -> list[PatternConfig]:
        """Get patterns for a specific data type.

        Args:
            data_type: Data type to filter by

        Returns:
            List of matching patterns
        """
        if isinstance(data_type, DataType):
            data_type = data_type.value
        return [p for p in self.get_all_patterns() if p.data_type == data_type]

    def validate_config(self, config: PatternConfigSchema) -> list[str]:
        """Validate a configuration.

        Args:
            config: Configuration to validate

        Returns:
            List of error messages
        """
        errors = []

        for pattern in config.get_all_patterns(include_disabled=True):
            # Validate regex
            try:
                re.compile(pattern.regex)
            except re.error as e:
                errors.append(f"Pattern '{pattern.pattern_id}': Invalid regex: {e}")

            # Validate examples
            example_results = pattern.validate_examples()
            for value, passed, message in example_results:
                if not passed:
                    errors.append(
                        f"Pattern '{pattern.pattern_id}': Example '{value}' failed: {message}"
                    )

            # Validate data type
            try:
                DataType(pattern.data_type)
            except ValueError:
                errors.append(
                    f"Pattern '{pattern.pattern_id}': Unknown data_type '{pattern.data_type}'"
                )

        return errors

    def enable_hot_reload(self, interval: float = 5.0) -> None:
        """Enable hot-reload of configuration files.

        Args:
            interval: Check interval in seconds
        """
        if self._hot_reload_enabled:
            return

        self._hot_reload_enabled = True
        self._stop_hot_reload.clear()

        def watch_loop() -> None:
            while not self._stop_hot_reload.wait(interval):
                self._check_for_changes()

        self._hot_reload_thread = threading.Thread(target=watch_loop, daemon=True)
        self._hot_reload_thread.start()

    def disable_hot_reload(self) -> None:
        """Disable hot-reload."""
        if not self._hot_reload_enabled:
            return

        self._stop_hot_reload.set()
        if self._hot_reload_thread:
            self._hot_reload_thread.join(timeout=2.0)
        self._hot_reload_enabled = False

    def _check_for_changes(self) -> None:
        """Check for file changes and reload if necessary."""
        with self._lock:
            paths_to_reload = []

            for path, mtime in list(self._file_mtimes.items()):
                try:
                    current_mtime = Path(path).stat().st_mtime
                    if current_mtime > mtime:
                        paths_to_reload.append(path)
                except OSError:
                    # File deleted or inaccessible
                    del self._configs[path]
                    del self._file_mtimes[path]

        for path in paths_to_reload:
            try:
                self.load_file(path)
            except Exception:
                pass  # Keep old config on reload failure

    def clear(self) -> None:
        """Clear all loaded configurations."""
        with self._lock:
            self._configs.clear()
            self._file_mtimes.clear()


# =============================================================================
# Pattern Registry
# =============================================================================


class PatternRegistry:
    """Global registry for custom patterns.

    Provides a singleton-like interface for pattern management.

    Example:
        # Register patterns
        registry = PatternRegistry()
        registry.load_file("patterns.yaml")

        # Use patterns
        for pattern in registry.get_patterns():
            if pattern.matches(value):
                print(f"Matched: {pattern.name}")
    """

    _instance: "PatternRegistry | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "PatternRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._loader = PatternConfigLoader()
        self._custom_patterns: dict[str, PatternConfig] = {}
        self._initialized = True

    @property
    def loader(self) -> PatternConfigLoader:
        """Access the internal loader."""
        return self._loader

    def load_file(self, path: str | Path) -> PatternConfigSchema:
        """Load patterns from file."""
        return self._loader.load_file(path)

    def load_directory(self, directory: str | Path) -> list[PatternConfigSchema]:
        """Load patterns from directory."""
        return self._loader.load_directory(directory)

    def register(self, pattern: PatternConfig) -> None:
        """Register a pattern programmatically.

        Args:
            pattern: Pattern to register
        """
        self._custom_patterns[pattern.pattern_id] = pattern

    def unregister(self, pattern_id: str) -> bool:
        """Unregister a pattern.

        Args:
            pattern_id: Pattern to unregister

        Returns:
            True if pattern was removed
        """
        if pattern_id in self._custom_patterns:
            del self._custom_patterns[pattern_id]
            return True
        return False

    def get_patterns(self, include_disabled: bool = False) -> list[PatternConfig]:
        """Get all registered patterns.

        Args:
            include_disabled: Whether to include disabled patterns

        Returns:
            List of patterns sorted by priority
        """
        all_patterns = self._loader.get_all_patterns(include_disabled)

        # Add custom patterns
        for p in self._custom_patterns.values():
            if p.enabled or include_disabled:
                all_patterns.append(p)

        # Remove duplicates and sort
        seen: dict[str, PatternConfig] = {}
        for p in all_patterns:
            if p.pattern_id not in seen or p.priority > seen[p.pattern_id].priority:
                seen[p.pattern_id] = p

        return sorted(seen.values(), key=lambda p: p.priority, reverse=True)

    def get_pattern(self, pattern_id: str) -> PatternConfig | None:
        """Get pattern by ID."""
        if pattern_id in self._custom_patterns:
            return self._custom_patterns[pattern_id]
        return self._loader.get_pattern(pattern_id)

    def match(self, value: str) -> list[PatternConfig]:
        """Find all patterns that match a value.

        Args:
            value: Value to match

        Returns:
            List of matching patterns (highest priority first)
        """
        matches = []
        for pattern in self.get_patterns():
            if pattern.matches(value):
                matches.append(pattern)
        return matches

    def match_first(self, value: str) -> PatternConfig | None:
        """Find the first (highest priority) matching pattern.

        Args:
            value: Value to match

        Returns:
            First matching pattern or None
        """
        for pattern in self.get_patterns():
            if pattern.matches(value):
                return pattern
        return None

    def clear(self) -> None:
        """Clear all patterns."""
        self._loader.clear()
        self._custom_patterns.clear()


# Global registry instance
pattern_registry = PatternRegistry()


# =============================================================================
# Default Patterns
# =============================================================================


DEFAULT_PATTERNS_YAML = r"""
version: "1.0"
name: "Default Patterns"
description: "Built-in patterns for common data types"

patterns:
  email:
    name: Email Address
    regex: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
    priority: 85
    data_type: email
    description: Standard email address format
    examples:
      - value: "user@example.com"
        should_match: true
      - value: "not-an-email"
        should_match: false

  url:
    name: URL
    regex: "^https?://[\\w.-]+(?:/[\\w./?%&=-]*)?$"
    priority: 80
    data_type: url
    description: HTTP/HTTPS URL
    examples:
      - value: "https://example.com/path"
        should_match: true

  uuid:
    name: UUID
    regex: "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    priority: 90
    data_type: uuid
    description: UUID v4 format
    examples:
      - value: "550e8400-e29b-41d4-a716-446655440000"
        should_match: true

  ip_v4:
    name: IPv4 Address
    regex: "^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
    priority: 85
    data_type: ip_address
    description: IPv4 address
    examples:
      - value: "192.168.1.1"
        should_match: true
      - value: "256.1.1.1"
        should_match: false

  iso_date:
    name: ISO Date
    regex: "^\\d{4}-\\d{2}-\\d{2}$"
    priority: 75
    data_type: date
    description: ISO 8601 date format
    examples:
      - value: "2024-12-25"
        should_match: true

  iso_datetime:
    name: ISO DateTime
    regex: "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?(?:Z|[+-]\\d{2}:\\d{2})?$"
    priority: 80
    data_type: datetime
    description: ISO 8601 datetime format
    examples:
      - value: "2024-12-25T10:30:00Z"
        should_match: true

groups:
  korean:
    name: Korean Patterns
    description: Patterns for Korean data formats
    priority_boost: 5
    patterns:
      korean_phone:
        name: Korean Phone Number
        regex: "^01[016789]-?\\d{3,4}-?\\d{4}$"
        priority: 90
        data_type: korean_phone
        examples:
          - value: "010-1234-5678"
            should_match: true
          - value: "01012345678"
            should_match: true

      korean_rrn:
        name: Korean RRN
        regex: "^\\d{6}-?[1-4]\\d{6}$"
        priority: 95
        data_type: korean_rrn
        description: Korean Resident Registration Number
        examples:
          - value: "900101-1234567"
            should_match: true

      korean_business_number:
        name: Korean Business Number
        regex: "^\\d{3}-\\d{2}-\\d{5}$"
        priority: 90
        data_type: korean_business_number
        examples:
          - value: "123-45-67890"
            should_match: true
"""


def load_default_patterns() -> None:
    """Load default patterns into the global registry."""
    pattern_registry.loader.load_from_string(DEFAULT_PATTERNS_YAML, "defaults")


# =============================================================================
# Convenience Functions
# =============================================================================


def load_patterns(path: str | Path) -> list[PatternConfig]:
    """Load patterns from a file.

    Args:
        path: Path to YAML file

    Returns:
        List of loaded patterns
    """
    config = pattern_registry.load_file(path)
    return config.get_all_patterns()


def load_patterns_directory(directory: str | Path) -> list[PatternConfig]:
    """Load patterns from a directory.

    Args:
        directory: Directory containing YAML files

    Returns:
        List of all loaded patterns
    """
    pattern_registry.load_directory(directory)
    return pattern_registry.get_patterns()


def register_pattern(
    pattern_id: str,
    regex: str,
    name: str | None = None,
    data_type: str = "string",
    priority: int = PatternPriority.MEDIUM,
    **kwargs: Any,
) -> PatternConfig:
    """Register a pattern programmatically.

    Args:
        pattern_id: Unique pattern identifier
        regex: Regular expression
        name: Human-readable name
        data_type: Inferred data type
        priority: Match priority
        **kwargs: Additional pattern options

    Returns:
        Created pattern config
    """
    pattern = PatternConfig(
        pattern_id=pattern_id,
        name=name or pattern_id,
        regex=regex,
        data_type=data_type,
        priority=priority,
        **kwargs,
    )
    pattern_registry.register(pattern)
    return pattern


def match_patterns(value: str) -> list[PatternConfig]:
    """Find patterns matching a value.

    Args:
        value: Value to match

    Returns:
        List of matching patterns
    """
    return pattern_registry.match(value)


def infer_type_from_patterns(value: str) -> DataType | None:
    """Infer data type from matching patterns.

    Args:
        value: Value to analyze

    Returns:
        Inferred DataType or None
    """
    pattern = pattern_registry.match_first(value)
    if pattern:
        return pattern.get_data_type()
    return None


def export_patterns(
    path: str | Path,
    patterns: list[PatternConfig] | None = None,
) -> None:
    """Export patterns to a YAML file.

    Args:
        path: Output file path
        patterns: Patterns to export (defaults to all registered)
    """
    if patterns is None:
        patterns = pattern_registry.get_patterns()

    config = {
        "version": "1.0",
        "name": "Exported Patterns",
        "patterns": {p.pattern_id: p.to_dict() for p in patterns},
    }

    content = _dump_yaml(config)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
