--8<-- "src/zodiax/numerics/overview.md"

## Reference

::: zodiax.numerics.expressions

::: zodiax.numerics.transforms

::: zodiax.numerics.operations

::: zodiax.numerics.links

::: zodiax.numerics.units

Array and State references are on the [arrays and state](stateful.md) page.


## Solving a matrix contraction

`solve_matmul(y, M)` is the minimum-norm least-squares counterpart to `matmul(x, M)`.
It supports sampled output axes and leading batch dimensions without requiring a
`Map` instance.

::: zodiax.numerics.transforms.solve_matmul
