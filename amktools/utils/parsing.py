import ast
import operator


# Based off https://stackoverflow.com/a/30134081
from typing import TypeVar, Type, Union

_operations = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
}


def _safe_eval(node, variables, functions):
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.Name):
        return variables[node.id]  # KeyError -> Unsafe variable
    elif isinstance(node, ast.BinOp):
        op = _operations[node.op.__class__]  # KeyError -> Unsafe operation
        left = _safe_eval(node.left, variables, functions)
        right = _safe_eval(node.right, variables, functions)
        if isinstance(node.op, ast.Pow):
            assert right < 100
        return op(left, right)
    elif isinstance(node, ast.Call):
        assert not node.keywords and not node.starargs and not node.kwargs
        assert isinstance(node.func, ast.Name), "Unsafe function derivation"
        func = functions[node.func.id]  # KeyError -> Unsafe function
        args = [_safe_eval(arg, variables, functions) for arg in node.args]
        return func(*args)

    assert False, "Unsafe operation"


# https://stackoverflow.com/a/20748308
# ast.literal_eval allows addition but bans multiplication.
# literal_eval(repr(1+2j)) == 1+2j
T = TypeVar("T")


def safe_eval(expr: Union[str, T], ret_type: Type[T], variables={}, functions={}) -> T:
    if not isinstance(expr, str):
        return check(expr, ret_type)

    node = ast.parse(expr, "<string>", "eval").body
    ret = _safe_eval(node, variables, functions)

    return check(ret, ret_type)


def check(val, ret_type):
    if ret_type and not isinstance(val, ret_type):
        raise TypeError(f"invalid expression {expr}, not type {ret_type}")
    return val
