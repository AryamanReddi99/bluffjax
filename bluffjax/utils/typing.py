from typing import Any, TypeAlias
from jaxtyping import Array, Float, Int, Bool, PyTree, PRNGKeyArray

# for type annotations only
Any: TypeAlias = Any
Array: TypeAlias = Array
FloatArray: TypeAlias = Float[Array, "..."]
IntArray: TypeAlias = Int[Array, "..."]
BoolArray: TypeAlias = Bool[Array, "..."]
PyTree: TypeAlias = PyTree
PRNGKeyArray: TypeAlias = PRNGKeyArray
