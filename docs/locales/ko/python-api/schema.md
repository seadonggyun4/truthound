# 스키마

Python API 사용에서 Schema을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 스키마 Class

Python API 사용에서 Container을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Definition

```python
from truthound.schema import Schema

@dataclass
class Schema:
    """Complete schema for a dataset."""

    columns: dict[str, ColumnSchema]  # Column name -> schema mapping
    row_count: int | None = None
    version: str = "1.0"
```

### Class Methods

#### `Schema.load()`

Load 스키마 from a YAML 파일.

```python
@classmethod
def load(cls, path: str | Path) -> Schema:
    """Load schema from YAML file."""
```

Python API 사용에서 Example을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
schema = Schema.load("schema.yaml")
```

#### `Schema.from_dict()`

Create 스키마 from a dictionary.

```python
@classmethod
def from_dict(cls, data: dict) -> Schema:
    """Create schema from dictionary."""
```

Python API 사용에서 Example을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
schema = Schema.from_dict({
    "version": "1.0",
    "row_count": 1000,
    "columns": {
        "id": {"name": "id", "dtype": "Int64", "nullable": False, "unique": True},
        "email": {"name": "email", "dtype": "String", "pattern": r"^[\w.+-]+@[\w-]+\.[\w.-]+$"},
    }
})
```

### Instance Methods

#### `Schema.save()`

Save 스키마 to a YAML 파일.

```python
def save(self, path: str | Path) -> None:
    """Save schema to YAML file."""
```

Python API 사용에서 Example을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
schema.save("schema.yaml")
```

#### `Schema.to_dict()`

Convert 스키마 to a dictionary.

```python
def to_dict(self) -> dict:
    """Convert to dictionary."""
```

#### `Schema.get_column_names()`

Get list of 컬럼 names.

```python
def get_column_names(self) -> list[str]:
    """Get list of column names."""
```

### Subscript Access

```python
# Access column schema by name
col_schema = schema["email"]

# Check if column exists
if "email" in schema:
    print("Email column exists")

# Iterate over column names
for col_name in schema:
    print(col_name)
```

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## ColumnSchema Class

Definition for a single 컬럼.

### Definition

```python
from truthound.schema import ColumnSchema

@dataclass
class ColumnSchema:
    """Schema definition for a single column."""

    name: str
    dtype: str
    nullable: bool = True
    unique: bool = False

    # Constraints
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[Any] | None = None
    pattern: str | None = None  # Regex pattern for strings
    min_length: int | None = None
    max_length: int | None = None

    # Statistics (learned from data)
    null_ratio: float | None = None
    unique_ratio: float | None = None
    mean: float | None = None
    std: float | None = None
    quantiles: dict[str, float] | None = None
```

### Fields

| Python API 사용에서 Field을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|------|---------|-------------|
| Python API 사용에서 `name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Required을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 name |
| Python API 사용에서 `dtype`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Required을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Data, Int64, String을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `nullable`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Allow을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `unique`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `False`, False을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Values을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `min_value`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Minimum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `max_value`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Maximum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `allowed_values`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `list`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Valid을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `pattern`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Regex을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `min_length`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Minimum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `max_length`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Maximum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `null_ratio`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Ratio을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `unique_ratio`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Ratio을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `mean`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Mean을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `std`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Standard을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `quantiles`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `dict`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Quantiles을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Supported Data Types

| Python API 사용에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-------------|
| Python API 사용에서 `Int8`, `Int16`, `Int32`, `Int64`, Int8, Int16, Int32, Int64을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Signed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `UInt8`, `UInt16`, `UInt32`, `UInt64`, UInt8, UInt16, UInt32, UInt64을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Unsigned을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `Float32`, `Float64`, Float32, Float64을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Floating을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `String`, `Utf8`, String, Utf8을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 UTF-8을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `Boolean`, Boolean을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 True/False을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `Date`, Date을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Date을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `Datetime`, Datetime을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Date을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `Time`, Time을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Time을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `Duration`, Duration을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Time을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Methods

#### `ColumnSchema.to_dict()`

```python
def to_dict(self) -> dict:
    """Convert to dictionary, excluding None values."""
```

#### `ColumnSchema.from_dict()`

```python
@classmethod
def from_dict(cls, data: dict) -> ColumnSchema:
    """Create ColumnSchema from dictionary."""
```

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## YAML Format

### Basic 스키마

```yaml
version: "1.0"
row_count: 10000
columns:
  id:
    name: id
    dtype: Int64
    nullable: false
    unique: true

  email:
    name: email
    dtype: String
    nullable: true
    pattern: "^[\\w.+-]+@[\\w-]+\\.[\\w.-]+$"

  age:
    name: age
    dtype: Int64
    nullable: true
    min_value: 0
    max_value: 150

  status:
    name: status
    dtype: String
    nullable: false
    allowed_values:
      - active
      - inactive
      - pending

  created_at:
    name: created_at
    dtype: Date
    nullable: false
```

### With Statistics (Learned 스키마)

```yaml
version: "1.0"
row_count: 50000
columns:
  user_id:
    name: user_id
    dtype: Int64
    nullable: false
    unique: true
    min_value: 1
    max_value: 50000
    null_ratio: 0.0
    unique_ratio: 1.0

  age:
    name: age
    dtype: Int64
    nullable: true
    min_value: 18
    max_value: 85
    null_ratio: 0.02
    unique_ratio: 0.068
    mean: 34.5
    std: 12.3
    quantiles:
      "25%": 25.0
      "50%": 33.0
      "75%": 42.0

  country:
    name: country
    dtype: String
    nullable: false
    allowed_values:
      - US
      - UK
      - CA
      - AU
    null_ratio: 0.0
    unique_ratio: 0.00008
```

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 사용 예시

### Learning and Saving 스키마

```python
import truthound as th

# Learn schema from reference data
schema = th.learn("reference_data.csv")

# Inspect learned schema
print(f"Columns: {schema.get_column_names()}")
print(f"Row count: {schema.row_count}")

for col_name, col_schema in schema.columns.items():
    print(f"\n{col_name}:")
    print(f"  Type: {col_schema.dtype}")
    print(f"  Nullable: {col_schema.nullable}")
    if col_schema.min_value is not None:
        print(f"  Range: [{col_schema.min_value}, {col_schema.max_value}]")
    if col_schema.allowed_values:
        print(f"  Allowed: {col_schema.allowed_values}")

# Save to file
schema.save("schema.yaml")
```

### Learning 스키마 from 데이터베이스

```python
import truthound as th
from truthound.datasources.sql import SQLiteDataSource, PostgreSQLDataSource

# Learn schema from SQLite database
source = SQLiteDataSource(database="mydb.db", table="users")
schema = th.learn(source=source)
schema.save("users_schema.yaml")

# Learn schema from PostgreSQL
source = PostgreSQLDataSource(
    table="orders",
    host="localhost",
    database="mydb",
    user="postgres",
)
schema = th.learn(source=source)
print(f"Learned schema with {len(schema.columns)} columns")
```

### Validating with 스키마

```python
import truthound as th

# Load schema
schema = Schema.load("schema.yaml")

# Validate new data against schema
report = th.check("new_data.csv", schema=schema)

if report.issues:
    print(f"Found {len(report.issues)} issues:")
    for issue in report.issues:
        print(f"  [{issue.severity.value}] {issue.column}: {issue.issue_type}")
```

### Programmatic 스키마 Creation

```python
from truthound.schema import Schema, ColumnSchema

schema = Schema(
    version="1.0",
    row_count=None,  # Unknown
    columns={
        "order_id": ColumnSchema(
            name="order_id",
            dtype="Int64",
            nullable=False,
            unique=True,
        ),
        "customer_email": ColumnSchema(
            name="customer_email",
            dtype="String",
            nullable=False,
            pattern=r"^[\w.+-]+@[\w-]+\.[\w.-]+$",
        ),
        "total": ColumnSchema(
            name="total",
            dtype="Float64",
            nullable=False,
            min_value=0.0,
        ),
        "status": ColumnSchema(
            name="status",
            dtype="String",
            nullable=False,
            allowed_values=["pending", "shipped", "delivered", "cancelled"],
        ),
    },
)

schema.save("orders_schema.yaml")
```

### 스키마 with Constraints Only (No Statistics)

```python
import truthound as th

# Learn schema without statistics
schema = th.learn("data.csv", infer_constraints=False)

# This gives you just types and basic properties
for col_name, col_schema in schema.columns.items():
    print(f"{col_name}: {col_schema.dtype}")
```

## 함께 보기

- Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 Validate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 CLI을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
