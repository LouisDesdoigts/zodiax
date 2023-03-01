# Filter Module

The filter module is a very lightweight wrapper for the `equinox` [filter functions](https://docs.kidger.site/equinox/api/filtering/transformations/). There are only two functions that are acutally modified, `filter_grad` and `filter_value_and_grad`. The rest of the functions are just wrappers for the `equinox` functions in order to have all filter functions accessible from the `filter` module.

The modified functions simply allow for either a set of arguments or a pytree to be used to select which parameters are differentiated with respect to. This is useful when you want to differentiate with respect to a subset of the parameters, rather than the full model.

Here is a simple example of the `filter_grad` function:

```python
import zodiax as zdx

class Linear(zdx.ExtendedBase):
    m : float
    b : float

    def __init__(self, m, b):
        self.m = np.asarray(m, float)
        self.b = np.asarray(b, float)
    
    def model(self, x):
        return self.m * x + self.b

# Make model
linear = Linear(1, 2)

# Differentiate with respect to the m parameter
args = 'm'
pytree_args = linear.get_args(args)

# Use the arguments to filter the gradient
@zdx.filter_grad(args)
def loss_fn(model, x):
    return np.abs(model.model(x) - x)

# Use the pytree to filter the gradient
@zdx.filter_grad(pytree_args)
def loss_fn(model, x):
    return np.square(model.model(x) - x)
```

The same syntax applies to the `filter_value_and_grad` function. All other functions are just wrappers for the `equinox` functions and therfore follow the same syntax.