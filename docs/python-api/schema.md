# Schema

Schema definition classes for data validation.

## Schema Class

Container for column schemas learned from data.

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

Load schema from a YAML file.

```python
@classmethod
def load(cls, path: str | Path) -> Schema:
    """Load schema from YAML file."""
```

**Example:**

```python
schema = Schema.load("schema.yaml")
```

#### `Schema.from_dict()`

Create schema from a dictionary.

```python
@classmethod
def from_dict(cls, data: dict) -> Schema:
    """Create schema from dictionary."""
```

**Example:**

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

Save schema to a YAML file.

```python
def save(self, path: str | Path) -> None:
    """Save schema to YAML file."""
```

**Example:**

```python
schema.save("schema.yaml")
```

#### `Schema.to_dict()`

Convert schema to a dictionary.

```python
def to_dict(self) -> dict:
    """Convert to dictionary."""
```

#### `Schema.get_column_names()`

Get list of column names.

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

---

## ColumnSchema Class

Definition for a single column.

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

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | Required | Column name |
| `dtype` | `str` | Required | Data type (e.g., "Int64", "String") |
| `nullable` | `bool` | `True` | Allow null values |
| `unique` | `bool` | `False` | Values must be unique |
| `min_value` | `float` | `None` | Minimum value (numeric) |
| `max_value` | `float` | `None` | Maximum value (numeric) |
| `allowed_values` | `list` | `None` | Valid values (categorical) |
| `pattern` | `str` | `None` | Regex pattern to match |
| `min_length` | `int` | `None` | Minimum string length |
| `max_length` | `int` | `None` | Maximum string length |
| `null_ratio` | `float` | `None` | Ratio of null values (learned) |
| `unique_ratio` | `float` | `None` | Ratio of unique values (learned) |
| `mean` | `float` | `None` | Mean value (numeric, learned) |
| `std` | `float` | `None` | Standard deviation (numeric, learned) |
| `quantiles` | `dict` | `None` | Quantiles {"25%", "50%", "75%"} (learned) |

### Supported Data Types

| Type | Description |
|------|-------------|
| `Int8`, `Int16`, `Int32`, `Int64` | Signed integers |
| `UInt8`, `UInt16`, `UInt32`, `UInt64` | Unsigned integers |
| `Float32`, `Float64` | Floating point |
| `String`, `Utf8` | UTF-8 string |
| `Boolean` | True/False |
| `Date` | Date without time |
| `Datetime` | Date with time |
| `Time` | Time without date |
| `Duration` | Time duration |

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

---

## YAML Format

### Basic Schema

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

### With Statistics (Learned Schema)

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

---

## Usage Examples

### Learning and Saving Schema

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

### Learning Schema from Database

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

### Validating with Schema

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

### Programmatic Schema Creation

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

### Schema with Constraints Only (No Statistics)

```python
import truthound as th

# Learn schema without statistics
schema = th.learn("data.csv", infer_constraints=False)

# This gives you just types and basic properties
for col_name, col_schema in schema.columns.items():
    print(f"{col_name}: {col_schema.dtype}")
```

## See Also

- [th.learn()](core-functions.md#thlearn) - Learn schema from data
- [th.check()](core-functions.md#thcheck) - Validate with schema
- [CLI: learn](../cli/core/learn.md) - CLI equivalent
