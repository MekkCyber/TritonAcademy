# Layer Normalization Backward Pass Derivation

## Forward Pass

First, let's establish the forward pass equations:

$$\mu = \frac{1}{n}\sum_{i=1}^n X_i$$

$$\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^n(X_i-\mu)^2}$$

$$\hat{X} = \frac{X - \mu}{\sigma}$$

$$Y = \gamma \odot \hat{X} + \beta$$

Where:
- $X$ is the input tensor
- $\mu$ is the mean (scalar)
- $\sigma$ is the standard deviation (scalar)
- $\hat{X}$ is the normalized input
- $\gamma$ and $\beta$ are learnable parameters
- $n$ is the feature dimension
- $\odot$ represents element-wise multiplication

## Backward Pass Derivation

We'll derive $\nabla_X$ (gradient with respect to input) given $\nabla_Y$ (gradient from the output).

### Step 1: Gradient from $Y$ to $\hat{X}$

Starting with $Y = \gamma \odot \hat{X} + \beta$

Taking the derivative with respect to $\hat{X}$:

$$ \nabla_{\hat{X}} = \frac{\partial \mathcal{L}}{\partial \hat{X}} = \frac{\partial \mathcal{L}}{\partial Y} \cdot \frac{\partial Y}{\partial \hat{X}} = \nabla_Y \odot \gamma$$

This means each element of the gradient with respect to $\hat{X}$ is the corresponding element of $\nabla_Y$ multiplied by the corresponding element of $\gamma$.

### Step 2: Gradient from $\hat{X}$ to $X$

Now we need to compute $\nabla_X$ given $\nabla_{\hat{X}}$, using the chain rule again : 

$$\nabla_X = \frac{\partial \hat{X}}{\partial X} \cdot \nabla_{\hat{X}}$$

We need to compute the gradient of $\hat{X}$ with respect to $X$. The normalized value is:
$$\hat{X} = \frac{X - \mu}{\sigma}$$

We need to account for how changes in $X$ affect $\hat{X}$ both directly and through $\mu$ and $\sigma$.

#### Component 1: Direct effect on $X$

For the direct effect (ignoring effects through $\mu$ and $\sigma$):
$$\frac{\partial \hat{X}}{\partial X}_{\text{direct}} = \frac{1}{\sigma}\mathbf{I}$$

Where $\mathbf{I}$ is the identity matrix.

#### Component 2: Effect through $\mu$

The mean $\mu = \frac{1}{n}\sum_{i=1}^n X_i$ depends on all elements of $X$.

For any element $X_j$:
$$\frac{\partial \mu}{\partial X_j} = \frac{1}{n}$$

The effect on $\hat{X}_i$ through $\mu$ is:
$$\frac{\partial \hat{X}_i}{\partial \mu} = -\frac{1}{\sigma}$$

Combining these:
$$\frac{\partial \hat{X}_i}{\partial X_j}_{\text{via }\mu} = \frac{\partial \hat{X}_i}{\partial \mu} \cdot \frac{\partial \mu}{\partial X_j} = -\frac{1}{\sigma} \cdot \frac{1}{n} = -\frac{1}{n\sigma}$$

#### Component 3: Effect through $\sigma$

The standard deviation $\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^n(X_i-\mu)^2}$ also depends on all elements of $X$.

First, let's compute $\frac{\partial \sigma}{\partial X_j}$:

$${2\sigma} \cdot \frac{\partial \sigma}{\partial X_j} = \frac{\partial}{\partial X_j}\left(\frac{1}{n}\sum_{i=1}^n(X_i-\mu)^2\right)$$

Which is : 
$$\frac{\partial \sigma}{\partial X_j} = \frac{1}{2\sigma} \cdot \frac{\partial}{\partial X_j}\left(\frac{1}{n}\sum_{i=1}^n(X_i-\mu)^2\right)$$

We need to account for both the direct effect on $(X_j-\mu)^2$ and the indirect effect through $\mu$ on all terms $(X_i-\mu)^2$.

The direct effect when $i = j$ is:
$$\frac{\partial}{\partial X_j}(X_j-\mu)^2 = 2(X_j-\mu) \cdot \left(1 - \frac{\partial \mu}{\partial X_j}\right) = 2(X_j-\mu) \cdot \left(1 - \frac{1}{n}\right)$$

The indirect effect through $\mu$ for each $i \neq j$ is:
$$\frac{\partial}{\partial X_j}(X_i-\mu)^2 = 2(X_i-\mu) \cdot \left(- \frac{\partial \mu}{\partial X_j}\right) = -2(X_i-\mu) \cdot \frac{1}{n}$$

Combining these and simplifying:
$$\frac{\partial \sigma}{\partial X_j} = \frac{1}{2\sigma} \cdot \frac{1}{n} \cdot \left(2(X_j-\mu)\left(1-\frac{1}{n}\right) - \sum_{i \neq j}2(X_i-\mu)\frac{1}{n}\right)$$

This further simplifies to:
$$\frac{\partial \sigma}{\partial X_j} = \frac{1}{n\sigma}(X_j-\mu)$$

because $\sum_{i=1}^n (X_i-\mu) = 0$ and that's because $\mu = \frac{1}{n}\sum_{i=1}^n X_i$.

Or in terms of $\hat{X}$:
$$\frac{\partial \sigma}{\partial X_j} = \frac{1}{n}\hat{X}_j$$

Now, the effect on $\hat{X}_i$ through $\sigma$ is:
$$\frac{\partial \hat{X}_i}{\partial \sigma} = -\frac{X_i-\mu}{\sigma^2} = -\frac{\hat{X}_i}{\sigma}$$

Combining these:
$$\frac{\partial \hat{X}_i}{\partial X_j}_{\text{via }\sigma} = \frac{\partial \hat{X}_i}{\partial \sigma} \cdot \frac{\partial \sigma}{\partial X_j} = -\frac{\hat{X}_i}{\sigma} \cdot \frac{1}{n}\hat{X}_j = -\frac{1}{n\sigma}\hat{X}_i\hat{X}_j$$

#### Combining All Components

Adding all three components together:

$$\nabla_{X_i} = \left(\frac{\partial \hat{X}}{\partial X}\right)_{i,:}\cdot \nabla_{\hat{X}} = \frac{1}{\sigma}\nabla_{\hat{X}_i} - \frac{1}{n\sigma}\sum_{j=1}^n \nabla_{\hat{X}_j} - \frac{\hat{X}_i}{n\sigma}\sum_{j=1}^n \nabla_{\hat{X}_j}\hat{X}_j$$

In vector notation:

$$\nabla_X = \frac{1}{\sigma}\nabla_{\hat{X}} - \frac{1}{n\sigma}\mathbf{1}\sum_{i=1}^n \nabla_{\hat{X}_i} - \frac{1}{n\sigma}\hat{X} \odot \left(\sum_{i=1}^n \nabla_{\hat{X}_i}\hat{X}_i\right)$$

Where $\mathbf{1}$ is a vector of ones.

Substituting $\nabla_{\hat{X}} = \nabla_Y \odot \gamma$:

$$\nabla_X = \frac{1}{\sigma}\left(\nabla_Y \odot \gamma - \frac{1}{n}\mathbf{1}\sum_{i=1}^n (\nabla_Y \odot \gamma)_i - \hat{X} \odot \frac{1}{n}\sum_{i=1}^n(\nabla_Y \odot \gamma \odot \hat{X})_i\right)$$

This can be written more compactly as:

$$\nabla_X = \frac{1}{\sigma}\left(\nabla_Y \odot \gamma - \left(\frac{1}{n}\hat{X} \cdot (\nabla_Y \odot \gamma)\right) \odot \hat{X} - \frac{1}{n}\nabla_Y \cdot \gamma \right)$$

This is the complete formula for the backward pass of layer normalization with respect to the input $X$.