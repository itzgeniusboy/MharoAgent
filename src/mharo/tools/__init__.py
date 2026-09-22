"""Tool registry — har tool REAL callable, Router/Engine ke liye bas
`registered()` return karta hai (tool name -> callable + JSON schema)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Tool:
    """Ek REAL tool: name + description + JSON schema + callable."""

    name: str
    description: str
    fn: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=dict)

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": [k for k, v in self.parameters.items()
                                 if getattr(v, "get", lambda _: False)("required")],
                },
            },
        }

    async def run(self, **kwargs) -> Any:
        res = self.fn(**kwargs)
        if hasattr(res, "__await__"):
            return await res
        return res


def add_tool(tool: Tool, registry: Optional[dict] = None) -> dict:
    """Register — returns registry (in-place default: global)."""
    reg = registry if registry is not None else REGISTRY
    reg[tool.name] = tool
    return reg


REGISTRY: dict[str, Tool] = {}


def registered() -> dict[str, Tool]:
    """Engine/Router ke liye instant tools ka dict."""
    return dict(REGISTRY)


# --- haal ke 2 real tools (zero-dep, tested) ---

def _calc(expression: str) -> float:
    import ast
    import operator

    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.USub: operator.neg, ast.Pow: operator.pow,
    }

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expr: {expression!r}")

    return float(ev(ast.parse(expression, mode="eval").body))


def _echo(text: str) -> str:
    return text


add_tool(Tool(
    name="calculator",
    description="Safe arithmetic evaluation of a math expression string.",
    fn=_calc,
    parameters={"expression": {"type": "string", "description": "math expression"}},
))
add_tool(Tool(
    name="echo",
    description="Returns the input text unchanged.",
    fn=_echo,
    parameters={"text": {"type": "string", "description": "text to echo"}},
))

__all__ = ["Tool", "REGISTRY", "add_tool", "registered"]