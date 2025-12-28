"""Generate API reference pages from source code.

This script automatically generates API documentation pages from
Python source files using mkdocstrings.
"""

from pathlib import Path
import mkdocs_gen_files

# Root source directory
SRC_DIR = Path("src/truthound")

# Navigation file for API reference
nav = mkdocs_gen_files.Nav()

# Modules to document
MODULES_TO_DOCUMENT = [
    # Core API
    "api",
    "schema",
    "drift",
    "report",
    "decorators",
    # Validators
    "validators",
    "validators/sdk",
    # Profiler
    "profiler",
    # Data Docs
    "datadocs",
    # Checkpoint
    "checkpoint",
    # Stores
    "stores",
    # ML
    "ml",
    # Lineage
    "lineage",
    # Realtime
    "realtime",
    # Plugins
    "plugins",
]

# Modules to skip
SKIP_PATTERNS = [
    "__pycache__",
    ".pyc",
    "_test.py",
    "test_",
]


def should_skip(path: Path) -> bool:
    """Check if a path should be skipped."""
    path_str = str(path)
    return any(pattern in path_str for pattern in SKIP_PATTERNS)


def get_module_doc_path(module_path: Path) -> Path:
    """Get the documentation path for a module."""
    # Remove src/truthound prefix
    relative = module_path.relative_to(SRC_DIR)

    # Convert to doc path
    doc_path = Path("api-reference") / relative.with_suffix(".md")

    return doc_path


def generate_module_page(py_file: Path, doc_path: Path) -> None:
    """Generate a documentation page for a Python module."""
    # Get the Python module path
    module_path = py_file.relative_to(SRC_DIR.parent)
    module_name = str(module_path.with_suffix("")).replace("/", ".")

    # Skip __init__ in the module name display
    display_name = module_name.replace(".__init__", "")

    # Generate the markdown content
    content = f"# {display_name.split('.')[-1]}\n\n"
    content += f"::: {module_name}\n"
    content += "    options:\n"
    content += "      show_source: true\n"
    content += "      show_root_heading: false\n"

    # Write the file
    with mkdocs_gen_files.open(doc_path, "w") as f:
        f.write(content)

    # Set edit path
    mkdocs_gen_files.set_edit_path(doc_path, py_file)


def process_directory(directory: Path, base_doc_path: Path) -> None:
    """Process a directory and generate docs for all Python files."""
    if not directory.exists():
        return

    for item in sorted(directory.iterdir()):
        if should_skip(item):
            continue

        if item.is_dir() and not item.name.startswith("_"):
            # Check for __init__.py
            init_file = item / "__init__.py"
            if init_file.exists():
                # Document the package
                doc_path = base_doc_path / item.name / "index.md"
                generate_module_page(init_file, doc_path)
                nav[item.name] = str(doc_path)

                # Process subdirectory
                process_directory(item, base_doc_path / item.name)

        elif item.is_file() and item.suffix == ".py":
            if item.name == "__init__.py":
                continue  # Already handled at directory level

            # Generate doc page
            doc_path = base_doc_path / item.with_suffix(".md").name
            generate_module_page(item, doc_path)
            nav[item.stem] = str(doc_path)


# Generate index page for API reference
index_content = """# API Reference

This section contains auto-generated API documentation from source code.

## Modules

"""

for module in MODULES_TO_DOCUMENT:
    module_path = SRC_DIR / module
    if module_path.is_dir():
        index_content += f"- [`truthound.{module.replace('/', '.')}`]({module}/index.md)\n"
    elif Path(str(module_path) + ".py").exists():
        index_content += f"- [`truthound.{module}`]({module}.md)\n"

with mkdocs_gen_files.open("api-reference/index.md", "w") as f:
    f.write(index_content)

# Process each module
for module in MODULES_TO_DOCUMENT:
    module_path = SRC_DIR / module

    if module_path.is_dir():
        # It's a package
        init_file = module_path / "__init__.py"
        if init_file.exists():
            doc_path = Path("api-reference") / module / "index.md"
            generate_module_page(init_file, doc_path)

            # Process submodules
            process_directory(module_path, Path("api-reference") / module)

    elif Path(str(module_path) + ".py").exists():
        # It's a single module
        py_file = Path(str(module_path) + ".py")
        doc_path = Path("api-reference") / f"{module}.md"
        generate_module_page(py_file, doc_path)

# Write navigation file
with mkdocs_gen_files.open("api-reference/SUMMARY.md", "w") as f:
    f.writelines(nav.build_literate_nav())
