# Cross Entropy

## Definition of Cross Entropy Loss
Cross Entropy Loss for a classification problem is defined as:

$$CE(x, class) = -\log(\text{softmax}(x)[class])$$

Where softmax is defined as:

$$\text{softmax}(x)[i] = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

## Expanded form of Cross Entropy Loss

$$CE(x, class) = -\log\left(\frac{e^{x_{class}}}{\sum_i e^{x_i}}\right)$$

$$CE(x, class) = -x_{class} + \log\left(\sum_i e^{x_i}\right)$$

Let's denote $z = \log\left(\sum_i e^{x_i}\right)$, which is the LogSumExp function.

## Compute gradients with respect to each logit

### Case 1: For the correct class (i = class)

$$\frac{\partial CE}{\partial x_{class}} = \frac{\partial}{\partial x_{class}}(-x_{class} + z)$$

$$\frac{\partial CE}{\partial x_{class}} = -1 + \frac{\partial z}{\partial x_{class}}$$

For the LogSumExp term:

$$\frac{\partial z}{\partial x_{class}} = \frac{\partial}{\partial x_{class}}\log\left(\sum_i e^{x_i}\right)$$

$$\frac{\partial z}{\partial x_{class}} = \frac{1}{\sum_i e^{x_i}} \cdot \frac{\partial}{\partial x_{class}}\left(\sum_i e^{x_i}\right)$$

$$\frac{\partial z}{\partial x_{class}} = \frac{1}{\sum_i e^{x_i}} \cdot e^{x_{class}}$$

$$\frac{\partial z}{\partial x_{class}} = \frac{e^{x_{class}}}{\sum_i e^{x_i}} = \text{softmax}(x)[class]$$

Substituting back:

$$\frac{\partial CE}{\partial x_{class}} = -1 + \text{softmax}(x)[class]$$

### Case 2: For other classes (i ≠ class)

$$\frac{\partial CE}{\partial x_i} = \frac{\partial}{\partial x_i}(-x_{class} + z)$$

$$\frac{\partial CE}{\partial x_i} = 0 + \frac{\partial z}{\partial x_i}$$

For the LogSumExp term same as before:

$$\frac{\partial z}{\partial x_i} = \frac{e^{x_i}}{\sum_j e^{x_j}} = \text{softmax}(x)[i]$$

For the correct class (i = class):

$$\frac{\partial CE}{\partial x_{class}} = -1 + \text{softmax}(x)[class]$$

For other classes (i ≠ class):

$$\frac{\partial CE}{\partial x_i} = \text{softmax}(x)[i]$$

## Generalize to a single formula
We can combine both cases into one formula:

$$\frac{\partial CE}{\partial x_i} = \text{softmax}(x)[i] - \mathbf{1}_{i=class}$$

Where $\mathbf{1}_{i=class}$ is an indicator function that equals 1 when i is the correct class and 0 otherwise.