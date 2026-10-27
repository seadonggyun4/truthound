"""Owner-scoped source imports that never reuse timestamp-based bytecode caches."""

import importlib.abc
import importlib.machinery
import importlib.util


class SourceLoader(importlib.abc.Loader):
    def __init__(self, path, failed):
        self.path = path
        self.failed = failed

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if module.__name__ in self.failed:
            raise ImportError("Source import already failed during this discovery")
        try:
            exec(compile(self.path.read_bytes(), str(self.path), "exec"), module.__dict__)
        except BaseException:
            self.failed.add(module.__name__)
            raise


class SourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.roots = {}
        self.failed = set()

    def find_spec(self, fullname, path=None, target=None):
        for prefix, root in self.roots.items():
            if not fullname.startswith(prefix + "."):
                continue
            parts = fullname[len(prefix) + 1 :].split(".")
            candidate = root.joinpath(*parts)
            package = candidate.is_dir()
            source = candidate / "__init__.py" if package else candidate.with_suffix(".py")
            if not source.resolve().is_relative_to(root):
                raise ImportError("Relative plugin import escapes its source root")
            if (
                any(
                    root.joinpath(*parts[:index]).is_symlink() for index in range(1, len(parts) + 1)
                )
                or source.is_symlink()
            ):
                raise ImportError("Relative plugin imports do not follow symlinks")
            if package and not source.exists():
                spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
                spec.submodule_search_locations = [str(candidate)]
                return spec
            if source.is_file():
                return importlib.util.spec_from_file_location(
                    fullname,
                    source,
                    loader=SourceLoader(source, self.failed),
                    submodule_search_locations=[str(candidate)] if package else None,
                )
            return None
        return None
