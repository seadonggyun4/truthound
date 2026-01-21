# truthound new validator

Create a custom validator with boilerplate code.

## Synopsis

```bash
truthound new validator <name> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `name` | Yes | Validator name (snake_case) |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `.` | Output directory |
| `--template` | `-t` | `basic` | Template type |
| `--author` | `-a` | None | Author name |
| `--description` | `-d` | None | Validator description |
| `--category` | `-c` | `custom` | Validator category |
| `--tests/--no-tests` | | `--tests` | Generate test code |
| `--docs/--no-docs` | | `--no-docs` | Generate documentation |
| `--severity` | `-s` | `MEDIUM` | Default severity level |
| `--pattern` | | None | Regex pattern (for pattern template) |
| `--min` | | None | Minimum value (for range template) |
| `--max` | | None | Maximum value (for range template) |

## Description

The `new validator` command generates validator boilerplate:

1. **Creates** validator class file
2. **Generates** test file (optional)
3. **Creates** documentation (optional)
4. **Sets up** proper structure

## Templates

| Template | Description | Best For |
|----------|-------------|----------|
| `basic` | Minimal validator structure | Simple checks |
| `column` | Column-level validator | Target column validation |
| `pattern` | Regex pattern matching | Format validation |
| `range` | Numeric range validator | Bounds checking |
| `comparison` | Cross-column comparison | Column relationships |
| `composite` | Multiple validator combination | Complex rules |
| `full` | All features included | Production validators |

## Examples

### Basic Validator

```bash
truthound new validator my_validator
```

Generated structure:
```
./my_validator/
├── __init__.py
├── validator.py
└── tests/
    └── test_validator.py
```

Generated code (`my_validator/validator.py`):
```python
from truthound.validators.base import Validator
from truthound.validators.report import ValidationIssue

class MyValidator(Validator):
    """Custom validator: my_validator"""

    name = "my_validator"
    severity = "MEDIUM"

    def validate(self, df, columns=None):
        issues = []
        # Add validation logic here
        return issues
```

### Column Validator

```bash
truthound new validator null_check --template column --tests
```

Generated code:
```python
from truthound.validators.base import Validator
from truthound.validators.report import ValidationIssue
import polars as pl

class NullCheckValidator(Validator):
    """Column-level validator: null_check"""

    name = "null_check"
    severity = "MEDIUM"

    def __init__(self, columns=None):
        self.columns = columns

    def validate(self, df, columns=None):
        issues = []
        target_columns = columns or self.columns or df.columns

        for col in target_columns:
            if col not in df.columns:
                continue
            # Add column validation logic here

        return issues
```

### Pattern Validator

```bash
truthound new validator email_format --template pattern --pattern "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
```

Generated code:
```python
import re
from truthound.validators.base import Validator
from truthound.validators.report import ValidationIssue

class EmailFormatValidator(Validator):
    """Pattern validator: email_format"""

    name = "email_format"
    severity = "MEDIUM"
    pattern = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

    def validate(self, df, columns=None):
        issues = []
        for col in columns or df.columns:
            if df[col].dtype != pl.Utf8:
                continue
            non_matching = df.filter(
                ~pl.col(col).str.contains(self.pattern.pattern)
            )
            if len(non_matching) > 0:
                issues.append(ValidationIssue(
                    validator=self.name,
                    column=col,
                    message=f"Found {len(non_matching)} values not matching pattern",
                    severity=self.severity,
                ))
        return issues
```

### Range Validator

```bash
truthound new validator percentage --template range --min 0 --max 100
```

Generated code:
```python
from truthound.validators.base import Validator
from truthound.validators.report import ValidationIssue
import polars as pl

class PercentageValidator(Validator):
    """Range validator: percentage"""

    name = "percentage"
    severity = "MEDIUM"
    min_value = 0
    max_value = 100

    def validate(self, df, columns=None):
        issues = []
        for col in columns or df.columns:
            if not df[col].dtype.is_numeric():
                continue
            out_of_range = df.filter(
                (pl.col(col) < self.min_value) | (pl.col(col) > self.max_value)
            )
            if len(out_of_range) > 0:
                issues.append(ValidationIssue(
                    validator=self.name,
                    column=col,
                    message=f"Found {len(out_of_range)} values outside range [{self.min_value}, {self.max_value}]",
                    severity=self.severity,
                ))
        return issues
```

### Full Featured Validator

```bash
truthound new validator customer_data \
  --template full \
  --docs \
  --author "John Doe" \
  --description "Validates customer data integrity" \
  --category compliance
```

Generated structure:
```
./customer_data/
├── __init__.py
├── validator.py
├── docs/
│   └── README.md
├── examples/
│   └── basic_usage.py
└── tests/
    └── test_validator.py
```

### Custom Output Directory

```bash
truthound new validator my_check -o ./validators/
```

Creates files in `./validators/my_check/` directory.

### Comparison Validator

```bash
truthound new validator date_order --template comparison
```

For validating relationships between columns (e.g., start_date < end_date).

### Composite Validator

```bash
truthound new validator full_address --template composite
```

For combining multiple validation rules into one.

## Template Comparison

| Template | Columns | Pattern | Range | Tests | Docs |
|----------|---------|---------|-------|-------|------|
| basic | - | - | - | Yes | - |
| column | Yes | - | - | Yes | - |
| pattern | Yes | Yes | - | Yes | - |
| range | Yes | - | Yes | Yes | - |
| comparison | Multi | - | - | Yes | - |
| composite | Multi | - | - | Yes | - |
| full | Yes | Optional | Optional | Yes | Yes |

## Use Cases

### 1. Business Rule Validation

```bash
truthound new validator order_status \
  --template pattern \
  --pattern "^(pending|confirmed|shipped|delivered|cancelled)$" \
  --description "Validate order status values"
```

### 2. Data Quality Check

```bash
truthound new validator price_range \
  --template range \
  --min 0.01 \
  --max 999999.99 \
  --severity HIGH
```

### 3. Compliance Validation

```bash
truthound new validator ssn_format \
  --template pattern \
  --pattern "^\d{3}-\d{2}-\d{4}$" \
  --category compliance \
  --docs
```

### 4. Multi-Column Validation

```bash
truthound new validator date_sequence \
  --template comparison \
  --description "Ensure created_at < updated_at"
```

## Generated Test Code

```python
import pytest
import polars as pl
from my_validator import MyValidator

class TestMyValidator:
    def test_validate_valid_data(self):
        df = pl.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
        })
        validator = MyValidator()
        issues = validator.validate(df)
        assert len(issues) == 0

    def test_validate_invalid_data(self):
        df = pl.DataFrame({
            "col1": [None, 2, 3],
        })
        validator = MyValidator()
        issues = validator.validate(df)
        # Add assertions based on your validation logic
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Generation error |
| 2 | Invalid arguments |

## Related Commands

- [`new reporter`](new-reporter.md) - Create custom reporter
- [`new plugin`](new-plugin.md) - Create plugin package
- [`new templates`](new-templates.md) - List available templates

## See Also

- [Scaffolding Overview](index.md)
- [Custom Validator Tutorial](../../tutorials/custom-validator.md)
- [Validators Guide](../../guides/validators.md)
