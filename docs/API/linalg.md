# Tree Linear Algebra

The `zodiax.linalg` module associates floating vectors and matrices with the
paths and shapes of their PyTree coordinates. Dtype and values remain on the
numerical array.
It provides serialisable views of columns, square blocks, diagonals, Jacobians,
Hessians, and Fisher information without storing JAX transformation closures. The
`zodiax.jacobian` and `zodiax.hessian` functions calculate and return the corresponding
typed results.

## Example

```python
import jax.numpy as jnp
import zodiax as zdx

parameters = {
    "position": jnp.array([1.0, 2.0], dtype=jnp.float32),
    "flux": jnp.array(3.0, dtype=jnp.float32),
}


def model(values):
    return values["flux"] * values["position"]


jacobian = zdx.jacobian(model, parameters)
hessian = zdx.hessian(lambda values: jnp.sum(model(values) ** 2), parameters)
fisher = zdx.Fisher.from_jacobian(jacobian)

print(jacobian)
# Jacobian(
#   matrix=f32[2,3],
#   layout=TreeLayout(paths=('flux', 'position'), shapes=((), (2,)))
# )

print(hessian)
# Hessian(
#   matrix=f32[3,3],
#   layout=TreeLayout(paths=('flux', 'position'), shapes=((), (2,)))
# )

print(fisher)
# Fisher(
#   matrix=f32[3,3],
#   layout=TreeLayout(paths=('flux', 'position'), shapes=((), (2,)))
# )

# Map the parameter axis, square blocks, or diagonal back to named leaves.
jacobian.columns(nested=True)
hessian.blocks(nested=True)
hessian.diagonal().nested_dict()
```

::: zodiax.linalg
