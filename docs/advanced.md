# Optimisation and Inference in Zodiax

This tutorial will cover how to use Zodiax to perform optimisation and inference on models. If you havent already please read the Using Zodiax (ADD LINK) tutorial first! We will start with a simple example of optimising a model using gradient descent, then show how to use the good deep mind gradient processing library [Optax](https://optax.readthedocs.io/en/latest/), then show how to use numpy to perform inference on the data, and finally show how to use derivates to calcaulte Fisher matrices.

Lets build a simple model that we can use to demonstrate optimising models in Zodiax. Lets start with a simple class that models a normal distribution:

```python
import zodiax as zdx
from jax import numpy as np, scipy as scp

class Normal(zdx.Base):
    mean      : np.ndarray
    scale     : np.ndarray
    amplitude : np.ndarray

    def __init__(self, mean, scale, amplitude):
        self.mean      = np.asarray(mean,      dtype=float)
        self.scale     = np.asarray(scale,     dtype=float)
        self.amplitude = np.asarray(amplitude, dtype=float)
    
    def model(self, width=10):
        xs = np.linspace(-width, width, 128)
        return self.amplitude * scp.stats.norm.pdf(xs, self.mean, self.scale)
```

This class simply models a normal distribution with a mean, scale and amplitude, and has a `.model()` method that is used to actually perform the calculation of the normal distribution.

!!! tip "Declaring attributes"
    When using `equinox` or `zodiax` the attibutes of the class must be
    declared in the class definition to determine the structure of the
    pytree that is created when the class is instantiated. This is done by
    adding a type hint to the attribute which can be any valid python type
    and is **not** type checked!

!!! info "`.model()` vs `.__call__()`"
    It is common in Equinox to not define a `.model()` method but rather a `.__call__()` method so that the instance of the class can be called like a function, ie:

    ```python
    normal = Normal(0, 1, 1)
    distribution = normal(10)
    ```

    This is a matter of personal preference, *however* when using Optax if you try to optimise a class that has a `.__call__()` method, you can thrown unhelpful errors. Becuase of this I recommend avoiding `.__call__()` methods and instead using `.model()` method.

Now we construct a class to store and model a set of multiple normals.

```python
class NormalSet(zdx.Base):
    normals : dict
    width   : np.ndarray

    def __init__(self, means, scales, amplitude, names, width=10):
        normals = {}
        for i in range(len(names)):
            normals[names[i]] = Normal(means[i], scales[i], amplitude[i])
        self.normals = normals
        self.width = np.asarray(width, dtype=float)
    
    def __getattr__(self, key):
        if key in self.normals.keys():
            return self.normals[key]
        else:
            raise AttributeError(f"{key} not in {self.normals.keys()}")
    
    def model(self):
        return np.array([normal.model(self.width) 
            for normal in self.normals.values()]).sum(0)

sources = NormalSet([-1., 2.], [1., 2.], [2., 4.], ['alpha', 'beta'])
```

This NormalSet class now lets us store an arbitrary number of `Normal` objects in a dictionary, and allows us to access them by their dictionary key. We can also model the sum of all the normals using the `.model()` method.

This is all the class set-up we need, now we can look at how to perform different types of optimisation and inference using this model.

!!! question "Whats with the `__getattr__` method?"
    This method eases working with nested structures and can be used to raise parameters from the lowst level of the class structure up to the top. In this example it allows us to access the individual `Normal` objects by their dictionary key. Using this method, these two lines are equivalent:

    ```python
    mu = sources.normals['alpha'].mean
    mu = sources.alpha.mean
    ```

    These methods can be chained together with multiple nested classes to make accessing parameters across large models much simpler!

    It is strongly reccomended that your classes have a `__getattr__` method implemented as it makes working with nested structures *much* easier! When doing so it is important to ensure that the method raises the correct error when the attribute is not found. This is done by raising an `AttributeError` with a message that includes the name of the attribute that was not found. 

---

## Simple Gradeint Descent

**Create some fake data**

Now lets take a lott at how we can recover the parameters of the model using gradient descent. To do this we need to create some fake data which we will do by modelling the normals and adding some noise.

Then we create a new instance of the model that we will use to recover the parameters from the data!

```python
import jax.random as jr
import matplotlib.pyplot as plt

# Make some data by adding some noise
key = jr.PRNGKey(0)
true_signal = sources.model()
data = true_signal + jr.normal(key, sources.model().shape)/50

# Create a model to initialise
initial_model = NormalSet([-3., 3.], [1., 1.], [2.5, 2.5], ['alpha', 'beta'])
```

??? abstract "Plotting code"
    ```python
    # Examine the data and the initial model
    plt.figure(figsize=(8, 4))
    xs = np.linspace(-sources.width, sources.width, len(data))
    plt.scatter(xs, data, s=10, label="Data")
    plt.plot(xs, true_signal, alpha=0.75, label="True Signal")
    plt.plot(xs, initial_model.model(), alpha=0.75, label="Initial model")
    plt.axhline(0, color="k", alpha=0.5)
    plt.title("Source Signal and data")
    plt.xlabel("Position")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()
    ```

![Initial Model](../assets/initial_model.png)

**Create a Loss Function**

Now lets create a loss function and use the gradients to perform a simple gradient descent recovery of the parameters:

```python
# Now lets construct a loss function
opt_parameters = [
    'alpha.mean', 'alpha.scale', 'alpha.amplitude',
    'beta.mean', 'beta.scale', 'beta.amplitude'
    ]
@zdx.filter_jit
@zdx.filter_value_and_grad(opt_parameters)
def loss_fn(model, data):
    return np.square(model.model() - data).sum()
```

!!! info "`@zdx.filter_jit`"
    The `@zdx.filter_jit` decorator is used to compile the function to XLA. This is not strictly necessary but it can speed up the function a huge amount. The reason we use this function and not the regular `@jax.jit` is that the Zodiax function will mark any non-optimisable parameters as *static*, such as strings. This allows us to create classes with extra meta-data that we don't have to manually declare as static! 

    Just-In-Time (jit) compiled functions are compiled the first time they are called, so we should call it once before using it in an optimistaion loop!

!!! info "`@zdx.filter_value_and_grad` & `opt_parameters`"
    Why did we use the `@zdx.filter_value_and_grad` decorator and what is the `opt_pararmeters` variable? This filter function operates similarly to the `@zdx.filter_jit` decorator by preventaing gradients being taken with respect to strings and lets us specifiy *exactly* what parameters we want to optimise. In this case we want to optimise the parameters of the individual normals, so we pass a list of of paths to those parameters!

```python
# Compile to XLA
model = initial_model
loss, grads = loss_fn(model, data)

# Optimise
losses = []
for i in range(200):
    loss, grads = loss_fn(model, data)
    step = grads.multiply(opt_parameters, -1e-2)
    model = zdx.apply_updates(model, step)
    losses.append(loss)
```

??? question "What is the `step`?"
    In gradient descent, we update our parameters towards the minimum of the loss function by taking a step in the opposite direction of the gradient (ie towards the minimum). The size of this step is controlled by the learning rate, which is a hyper-parameter that we have to tune. In this case we have set the learning rate to `1e-2` and applied this learning rate to all of the parameters in the model.

How easy was that! Now lets examine the results:

??? abstract "Plotting code"
    ```python
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(np.log10(np.array(losses)))
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Log10 Loss")

    plt.subplot(1, 2, 2)
    plt.scatter(xs, data, s=10, label='Data')
    plt.plot(xs, model.model(), alpha=0.75, label='Recovered Model',  c='tab:green')
    plt.axhline(0, color="k", alpha=0.5)
    plt.title("Data and Recovered Signal")
    plt.xlabel("Position")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()
    ```

![GD basic](../assets/gd_fit.png)

---

## Optax

One of the other benfeits of using Zodiax is that our objects natively integrate in to the Jax optmisation ecosystem. For example we can use the Google DeepMind gradient processing libaray [Optax](https://optax.readthedocs.io/en/latest/) in order to gain access to a series of gradient optimisation algorithms.

We can re-use the loss function from above, so lets have a look how we can use some Optax optimisers:

```python
import optax

# Get optax objcets
model = initial_model
optimiser, state = zdx.get_optimiser(model, opt_parameters, optax.adam(1e-1))

losses = []
for i in range(200):
    loss, grads = loss_fn(model, data)
    step, state = optimiser.update(grads, state)
    model = zdx.apply_updates(model, step)
    losses.append(loss)
```

!!! info "`zdx.get_optimiser(model, parameters, optimisers)`"
    The `zdx.get_optimiser` function takes a model, a list of parameters to optimise and list of optimisers for each of the those parameters and returns an optax optimiser and an optax state object. This convenience function simply ensures that the we format our model correctly and map our optimisers correctly for Optax! These objects together are used to implement more complex gradient descent algorithms such as Adam, RMSProp, etc.

Easy! Lets examine the results

??? abstract "Plotting code"

    ```python
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(np.log10(np.array(losses)))
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Log10 Loss")

    plt.subplot(1, 2, 2)
    plt.scatter(xs, data, s=10, label='Data')
    plt.plot(xs, model.model(), alpha=0.75, label='Recovered Model',  c='tab:green')
    plt.axhline(0, color="k", alpha=0.5)
    plt.title("Data and Recovered Signal")
    plt.xlabel("Position")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()
    ```

![Optax](../assets/optax_fit.png)

---

## Fisher Inference

TODO: Move the bayesian module in Zodiax??

The differentiable nature of Zodiax objects also allows us to perform inference on the parameters of our model. The [Laplace approximation](https://en.wikipedia.org/wiki/Laplace%27s_approximation) assumes that the posterior distribution of our model parameters is a gaussian distribution centred on the maximum likelihood estimate of the parameters. Luckily we can use autodiff to calculate the hessian of the log likelihood function and invert it to get the covariance matrix of the posterior distribution!

??? info "Fisher and Covariance Matrices"
    The covariance matrix $\vec{\Sigma}$ describes the covariance between the parameters of a model. Under the laplace approximation (add link), we can calculate the covariance matrix using autodiff:

    $$
    \vec{\Sigma}(\vec{X}) = - \Big[ \nabla \nabla \text{ln} \Big(  
        \mathcal {L}(\vec{X}) \rho(\vec{X})
    \Big) \Big]^{-1}
    $$

    where $\mathcal{L}$ is the likelihood function and $\rho$ is the prior function. In this example we will assume a flat prior, so $\rho(\vec{X}) = 1$.

```python
# Define the paramters we want to marginalise over
parameters = ['alpha.mean',      'beta.mean', 
              'alpha.scale',     'beta.scale', 
              'alpha.amplitude', 'beta.amplitude']

# Define Likelihod function
def chi2(X, model, data, noise=1):
    signal = perturb(X, model).model()
    return np.log10(np.square((signal - data) / noise).sum())

# Define Perturbation function
def perturb(X, model):
    for parameter, x in zip(parameters, X):
        model = model.add(parameter, x)
    return model

# Calcuate the Fisher and Covariance matrix, standard deviation
X = np.zeros(len(parameters))
fisher_information = jax.hessian(chi2)(X, model, data)
covaraince_matrix = np.linalg.inv(fisher_information)
deviations = np.abs(np.diag(covaraince_matrix))**0.5
```

Lets examine the results:

??? abstract "Plotting code"
    ```python
    true_values = sources.get(parameters)
    recoverd_parameters = model.get(parameters)

    formatted = [r"$\alpha_\mu$",    r"$\beta_\mu$",
                r"$\alpha_\sigma$", r"$\beta_\sigma$",
                r"$\alpha_A$",      r"$\beta_A$"]

    plt.figure(figsize=(8, 4))
    xs = np.arange(len(parameters))
    plt.bar(xs, true_values, tick_label=formatted, width=0.3, label='True')
    plt.bar(xs+0.33, recoverd_parameters, tick_label=formatted, yerr=deviations,
        width=0.33, label='Recovered', capsize=3)
    plt.plot([], c='k', label='1-sigma')
    plt.axhline(0, color="k", alpha=0.5)
    plt.legend(loc=2)
    plt.xlabel("Parameter")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig("fisher_fit")
    plt.show()
    ```

![Fisher](../assets/fisher_fit.png)

Fantastic, this method gives us a great way to estimate the uncertainty in our recovered model parameters!

---

## Numpyro

[Numpyro](https://num.pyro.ai/en/latest/index.html) is a probabailistic programming library that allows us to perform efficient posterior sampling algorithms. Numpyro is also built in Jax and so is designed to that take advantage of differentiability in order to perform extremely high-dimensional inference!

Lets see how Zodiax can be integrated with Numpyro to perform posterior sampling on our model parameters.

```python
import numpyro as npy
import numpyro.distributions as dist
import chainconsumer as cc

def sampling_fn(data, model):
    paths = ["alpha.mean",     "beta.mean", 
            "alpha.scale",     "beta.scale",
            "alpha.amplitude", "beta.amplitude"]

    # Define priors
    values = [npy.sample(r"\alpha_\mu",    dist.Normal(0, 5)), 
              npy.sample(r"\beta_\mu",     dist.Normal(0, 5)),
              npy.sample(r"\alpha_\sigma", dist.HalfNormal(5)), 
              npy.sample(r"\beta_\sigma",  dist.HalfNormal(5)),
              npy.sample(r"\alpha_A",      dist.Normal(0, 5)), 
              npy.sample(r"\beta_A",       dist.Normal(0, 5))]
    
    # Sample from the posterior distribution
    with npy.plate("data", len(data)):
        model_sampler = dist.Normal(
            model.set_and_call(paths, values, "model")
            )
        return npy.sample("Sampler", model_sampler, obs=data)
```

Numpyo requires a 'sampling' function where you assign priors to your parameters and then sample from the posterior distribution. The syntax for this can be seen above. We then sample the data using a 'plate` and define a likelihood which in this case is a normal. The `set_and_call` function is a Zodiax function that allows us to update the model parameters and then return call some method of that class. This is the function that ultimately allows a simple interface with Numpyro.

We then need to define our sampler which in this case is the No U-Turn Sampler (NUTS). NUTS is a variant of Hamiltonian Monte Carlo (HMC) that is designed to be more efficient and robust, and takes advantage of gradients to allow high dimensional inference.

```python
# Using the model above, we can now sample from the posterior distribution
# using the No U-Turn Sampler (NUTS).
sampler = npy.infer.MCMC(
    npy.infer.NUTS(sampling_fn),
    num_warmup=5000,
    num_samples=5000,
)
%time sampler.run(jr.PRNGKey(0), data, model)
```

Fantastic now lets have a look at our posterior distributions!

??? abstract "Plotting code"
    ```python
    chain = cc.ChainConsumer()
    chain.add_chain(sampler.get_samples())
    chain.configure(
        serif=True, shade=True, bar_shade=True, shade_alpha=0.2, spacing=1.0, max_ticks=3
    )

    fig = chain.plotter.plot()
    fig.set_size_inches((15, 15))
    ```

![Numpyro](../assets/hmc_fit.png)

---

## Conclusion

This tutorial has covered the basics of using Zodiax for scientific programming. We have seen how we can construct fully differentiable object-oriented classes using Zodiax. We have seen how we can optimise parameters, estimate their undcertainties and how to sample from posterier distributions!