"""Rule evaluation engines for Python expressions and Jinja2 templates.

This module provides flexible rule evaluation through:
- ExpressionEngine: Safe Python expression evaluation with AST-based security
- Jinja2Engine: Template-based condition rendering with sandboxed execution
- RuleEngine: Unified facade for both engines

Security:
    - Python expressions are parsed and validated using AST
    - Only whitelisted operations and names are allowed
    - Jinja2 uses sandboxed environment with restricted features
"""

from __future__ import annotations

import ast
import logging
import operator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from truthound.checkpoint.routing.base import RouteContext, RoutingRule

logger = logging.getLogger(__name__)


# Safe operators for expression evaluation
SAFE_OPERATORS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.And: lambda a, b: a and b,
    ast.Or: lambda a, b: a or b,
    ast.Not: operator.not_,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Invert: operator.invert,
}

# Safe built-in functions
SAFE_BUILTINS: dict[str, Callable[..., Any]] = {
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "any": any,
    "all": all,
    "sorted": sorted,
    "reversed": lambda x: list(reversed(x)),
    "enumerate": lambda x: list(enumerate(x)),
    "zip": lambda *args: list(zip(*args)),
    "range": range,
    "isinstance": isinstance,
    "hasattr": hasattr,
    "getattr": getattr,
}


class ExpressionSecurityError(Exception):
    """Raised when an expression contains unsafe operations."""

    pass


class ExpressionEvaluationError(Exception):
    """Raised when expression evaluation fails."""

    pass


class SafeExpressionVisitor(ast.NodeVisitor):
    """AST visitor that validates expression safety.

    This visitor walks the AST and ensures only safe operations
    and names are used in expressions.
    """

    # Allowed AST node types
    ALLOWED_NODES = frozenset({
        ast.Module,
        ast.Expr,
        ast.Expression,
        # Literals
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Set,
        # Variables
        ast.Name,
        ast.Load,
        ast.Store,
        # Binary/Unary operations
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        # Binary operators
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.LShift,
        ast.RShift,
        ast.BitOr,
        ast.BitXor,
        ast.BitAnd,
        # Comparison operators
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
        # Boolean operators
        ast.And,
        ast.Or,
        ast.Not,
        # Unary operators
        ast.UAdd,
        ast.USub,
        ast.Invert,
        # Subscript
        ast.Subscript,
        ast.Slice,
        # Attribute access (restricted)
        ast.Attribute,
        # Conditionals
        ast.IfExp,
        # Function calls (restricted)
        ast.Call,
        # Comprehensions (restricted)
        ast.ListComp,
        ast.DictComp,
        ast.SetComp,
        ast.GeneratorExp,
        ast.comprehension,
    })

    # Forbidden attribute names
    FORBIDDEN_ATTRS = frozenset({
        "__class__",
        "__bases__",
        "__mro__",
        "__subclasses__",
        "__code__",
        "__globals__",
        "__builtins__",
        "__import__",
        "__reduce__",
        "__reduce_ex__",
        "__getstate__",
        "__setstate__",
        "__init__",
        "__new__",
        "__del__",
        "__dict__",
        "__doc__",
    })

    def __init__(self, allowed_names: set[str]) -> None:
        """Initialize the visitor.

        Args:
            allowed_names: Set of allowed variable names
        """
        self.allowed_names = allowed_names
        self.errors: list[str] = []

    def visit(self, node: ast.AST) -> None:
        """Visit a node and check safety."""
        if type(node) not in self.ALLOWED_NODES:
            self.errors.append(
                f"Forbidden AST node type: {type(node).__name__}"
            )
            return

        super().visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Check variable name access."""
        if node.id not in self.allowed_names:
            self.errors.append(f"Forbidden name access: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check attribute access."""
        if node.attr in self.FORBIDDEN_ATTRS:
            self.errors.append(f"Forbidden attribute access: {node.attr}")
        if node.attr.startswith("_"):
            self.errors.append(f"Private attribute access forbidden: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls."""
        # Only allow calls to known safe functions
        if isinstance(node.func, ast.Name):
            if node.func.id not in SAFE_BUILTINS and node.func.id not in self.allowed_names:
                self.errors.append(f"Forbidden function call: {node.func.id}")
        self.generic_visit(node)


@dataclass
class ExpressionEngine:
    """Safe Python expression evaluation engine.

    Evaluates Python expressions against a context dictionary using
    AST-based validation for security.

    Example:
        >>> engine = ExpressionEngine()
        >>> context = {"critical_issues": 5, "status": "failure"}
        >>> engine.evaluate("critical_issues > 0 and status == 'failure'", context)
        True

    Attributes:
        cache_compiled: Whether to cache compiled expressions
        max_expression_length: Maximum allowed expression length
    """

    cache_compiled: bool = True
    max_expression_length: int = 1000
    _cache: dict[str, ast.Expression] = field(default_factory=dict, repr=False)

    def validate(self, expression: str, allowed_names: set[str]) -> list[str]:
        """Validate an expression for safety.

        Args:
            expression: The Python expression to validate.
            allowed_names: Set of allowed variable names.

        Returns:
            List of validation errors (empty if valid).
        """
        if len(expression) > self.max_expression_length:
            return [f"Expression too long: {len(expression)} > {self.max_expression_length}"]

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            return [f"Syntax error: {e}"]

        # Add safe builtins to allowed names
        full_allowed = allowed_names | set(SAFE_BUILTINS.keys())

        visitor = SafeExpressionVisitor(full_allowed)
        visitor.visit(tree)

        return visitor.errors

    def compile(self, expression: str) -> ast.Expression:
        """Compile an expression to AST.

        Args:
            expression: The Python expression to compile.

        Returns:
            Compiled AST.

        Raises:
            ExpressionSecurityError: If expression is invalid or unsafe.
        """
        if self.cache_compiled and expression in self._cache:
            return self._cache[expression]

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            raise ExpressionSecurityError(f"Invalid expression syntax: {e}") from e

        if self.cache_compiled:
            self._cache[expression] = tree

        return tree

    def evaluate(
        self,
        expression: str,
        context: dict[str, Any],
        *,
        validate: bool = True,
    ) -> Any:
        """Evaluate an expression against a context.

        Args:
            expression: The Python expression to evaluate.
            context: Dictionary of variables available in the expression.
            validate: Whether to validate the expression for safety.

        Returns:
            The result of evaluating the expression.

        Raises:
            ExpressionSecurityError: If expression is unsafe.
            ExpressionEvaluationError: If evaluation fails.
        """
        if validate:
            errors = self.validate(expression, set(context.keys()))
            if errors:
                raise ExpressionSecurityError(
                    f"Expression validation failed: {'; '.join(errors)}"
                )

        try:
            tree = self.compile(expression)
            code = compile(tree, "<expression>", "eval")

            # Create safe namespace
            namespace = dict(context)
            namespace.update(SAFE_BUILTINS)

            return eval(code, {"__builtins__": {}}, namespace)
        except ExpressionSecurityError:
            raise
        except Exception as e:
            raise ExpressionEvaluationError(
                f"Expression evaluation failed: {e}"
            ) from e

    def clear_cache(self) -> None:
        """Clear the compiled expression cache."""
        self._cache.clear()


@dataclass
class Jinja2Engine:
    """Jinja2 template-based condition evaluation engine.

    Uses Jinja2 templates for more complex condition expressions
    with a sandboxed environment for security.

    Example:
        >>> engine = Jinja2Engine()
        >>> context = {"tags": {"env": "prod"}, "critical_issues": 5}
        >>> engine.evaluate(
        ...     "{{ tags.get('env') == 'prod' and critical_issues > 0 }}",
        ...     context
        ... )
        True

    Attributes:
        autoescape: Whether to auto-escape HTML
        cache_templates: Whether to cache compiled templates
    """

    autoescape: bool = False
    cache_templates: bool = True
    _env: Any = field(default=None, repr=False)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Initialize the Jinja2 environment."""
        self._init_env()

    def _init_env(self) -> None:
        """Initialize the sandboxed Jinja2 environment."""
        try:
            from jinja2.sandbox import SandboxedEnvironment
        except ImportError:
            logger.warning(
                "Jinja2 not installed, template evaluation unavailable"
            )
            return

        self._env = SandboxedEnvironment(
            autoescape=self.autoescape,
        )

        # Add safe globals
        self._env.globals.update({
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "range": range,
            "any": any,
            "all": all,
        })

    def _ensure_env(self) -> None:
        """Ensure environment is initialized."""
        if self._env is None:
            self._init_env()
            if self._env is None:
                raise ImportError(
                    "Jinja2 is required for template evaluation. "
                    "Install it with: pip install jinja2"
                )

    def render(self, template: str, context: dict[str, Any]) -> str:
        """Render a template with the given context.

        Args:
            template: The Jinja2 template string.
            context: Dictionary of variables for template rendering.

        Returns:
            Rendered template string.
        """
        self._ensure_env()

        if self.cache_templates and template in self._cache:
            compiled = self._cache[template]
        else:
            compiled = self._env.from_string(template)
            if self.cache_templates:
                self._cache[template] = compiled

        return compiled.render(**context)

    def evaluate(self, template: str, context: dict[str, Any]) -> bool:
        """Evaluate a template condition.

        The template should render to a boolean-like value.
        Empty strings, "false", "0", "none" are treated as False.

        Args:
            template: The Jinja2 template for the condition.
            context: Dictionary of variables for evaluation.

        Returns:
            Boolean result of the condition.
        """
        result = self.render(template, context).strip().lower()

        if result in ("", "false", "0", "none", "null", "no"):
            return False
        if result in ("true", "1", "yes"):
            return True

        # Try to parse as expression
        try:
            return bool(result)
        except (ValueError, TypeError):
            return bool(result)

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._cache.clear()


@dataclass
class ExpressionRule:
    """A routing rule based on Python expression evaluation.

    Example:
        >>> rule = ExpressionRule("critical_issues > 0 and status == 'failure'")
        >>> context = RouteContext(critical_issues=5, status="failure", ...)
        >>> rule.evaluate(context)
        True
    """

    expression: str
    _engine: ExpressionEngine = field(
        default_factory=ExpressionEngine, repr=False
    )
    _description: str | None = None

    def evaluate(self, context: "RouteContext") -> bool:
        """Evaluate the expression against the context.

        Args:
            context: The routing context.

        Returns:
            True if the expression evaluates to truthy.
        """
        ctx_dict = context.to_dict()
        result = self._engine.evaluate(self.expression, ctx_dict)
        return bool(result)

    @property
    def description(self) -> str:
        """Get rule description."""
        return self._description or f"Expression: {self.expression}"


@dataclass
class Jinja2Rule:
    """A routing rule based on Jinja2 template evaluation.

    Example:
        >>> rule = Jinja2Rule("{{ tags.get('env') == 'prod' }}")
        >>> context = RouteContext(tags={"env": "prod"}, ...)
        >>> rule.evaluate(context)
        True
    """

    template: str
    _engine: Jinja2Engine = field(default_factory=Jinja2Engine, repr=False)
    _description: str | None = None

    def evaluate(self, context: "RouteContext") -> bool:
        """Evaluate the template against the context.

        Args:
            context: The routing context.

        Returns:
            True if the template evaluates to truthy.
        """
        ctx_dict = context.to_dict()
        return self._engine.evaluate(self.template, ctx_dict)

    @property
    def description(self) -> str:
        """Get rule description."""
        return self._description or f"Template: {self.template}"


class RuleEngine:
    """Unified facade for expression and template evaluation.

    Provides a single interface for both Python expressions and
    Jinja2 templates, automatically detecting the appropriate engine.

    Example:
        >>> engine = RuleEngine()
        >>> # Python expression
        >>> rule1 = engine.create_rule("critical_issues > 0")
        >>> # Jinja2 template (detected by {{ }})
        >>> rule2 = engine.create_rule("{{ tags.env == 'prod' }}")
    """

    def __init__(self) -> None:
        """Initialize both engines."""
        self._expr_engine = ExpressionEngine()
        self._jinja2_engine = Jinja2Engine()

    def is_template(self, expression: str) -> bool:
        """Check if expression is a Jinja2 template.

        Args:
            expression: The expression to check.

        Returns:
            True if the expression contains Jinja2 syntax.
        """
        return "{{" in expression or "{%" in expression

    def create_rule(
        self,
        expression: str,
        description: str | None = None,
    ) -> "RoutingRule":
        """Create a rule from an expression.

        Automatically selects the appropriate engine based on
        expression syntax.

        Args:
            expression: Python expression or Jinja2 template.
            description: Optional human-readable description.

        Returns:
            A RoutingRule instance.
        """
        if self.is_template(expression):
            return Jinja2Rule(
                template=expression,
                _engine=self._jinja2_engine,
                _description=description,
            )
        return ExpressionRule(
            expression=expression,
            _engine=self._expr_engine,
            _description=description,
        )

    def evaluate(
        self,
        expression: str,
        context: "RouteContext",
    ) -> bool:
        """Evaluate an expression against a context.

        Args:
            expression: Python expression or Jinja2 template.
            context: The routing context.

        Returns:
            Boolean result of evaluation.
        """
        rule = self.create_rule(expression)
        return rule.evaluate(context)

    def validate_expression(
        self,
        expression: str,
        allowed_names: set[str] | None = None,
    ) -> list[str]:
        """Validate an expression for safety.

        Args:
            expression: The expression to validate.
            allowed_names: Optional set of allowed variable names.

        Returns:
            List of validation errors (empty if valid).
        """
        if self.is_template(expression):
            # Jinja2 templates are sandboxed, basic validation only
            try:
                self._jinja2_engine._ensure_env()
                self._jinja2_engine._env.from_string(expression)
                return []
            except Exception as e:
                return [f"Template validation failed: {e}"]

        # Use RouteContext fields as default allowed names
        if allowed_names is None:
            from truthound.checkpoint.routing.base import RouteContext
            import dataclasses

            allowed_names = {
                f.name for f in dataclasses.fields(RouteContext)
            }

        return self._expr_engine.validate(expression, allowed_names)

    def clear_caches(self) -> None:
        """Clear all caches."""
        self._expr_engine.clear_cache()
        self._jinja2_engine.clear_cache()
