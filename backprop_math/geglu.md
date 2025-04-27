# Derivative of GeLU

## Exact Derivative

Starting with:

$$f = \frac{1}{2} \cdot x \cdot \left(1 + \text{erf}\left(\frac{1}{\sqrt{2}} \cdot x\right)\right)$$

Using the product rule: 

$$\frac{d}{dx}[u(x) \cdot v(x)] = u'(x) \cdot v(x) + u(x) \cdot v'(x)$$

Let:

- $u(x) = \frac{1}{2} \cdot x$
- $v(x) = 1 + \text{erf}\left(\frac{1}{\sqrt{2}} \cdot x\right)$

Step 1: Find $u'(x)$

$$u'(x) = \frac{d}{dx}\left[\frac{1}{2} \cdot x\right] = \frac{1}{2}$$

Step 2: Find $v'(x)$
We need the chain rule here. The derivative of erf(x) is 

$$\frac{2}{\sqrt{\pi}} \cdot e^{-x^2}$$

$$v'(x) = \frac{d}{dx}\left[1 + \text{erf}\left(\frac{1}{\sqrt{2}} \cdot x\right)\right] = \frac{d}{dx}\left[\text{erf}\left(\frac{1}{\sqrt{2}} \cdot x\right)\right]$$

Using the chain rule with $g(x) = \frac{1}{\sqrt{2}} \cdot x$:

$$v'(x) = \frac{2}{\sqrt{\pi}} \cdot e^{-\left(\frac{1}{\sqrt{2}} \cdot x\right)^2} \cdot \frac{d}{dx}\left[\frac{1}{\sqrt{2}} \cdot x\right]$$

$$v'(x) = \frac{2}{\sqrt{\pi}} \cdot e^{-\frac{x^2}{2}} \cdot \frac{1}{\sqrt{2}}$$

$$v'(x) = \frac{2}{\sqrt{\pi}} \cdot \frac{1}{\sqrt{2}} \cdot e^{-\frac{x^2}{2}}$$

$$v'(x) = \frac{2}{\sqrt{2\pi}} \cdot e^{-\frac{x^2}{2}}$$

Step 3: Apply the product rule

$$\frac{df}{dx} = u'(x) \cdot v(x) + u(x) \cdot v'(x)$$

$$\frac{df}{dx} = \frac{1}{2} \cdot \left(1 + \text{erf}\left(\frac{1}{\sqrt{2}} \cdot x\right)\right) + \frac{1}{2} \cdot x \cdot \frac{2}{\sqrt{2\pi}} \cdot e^{-\frac{x^2}{2}}$$

$$\frac{df}{dx} = \frac{1}{2} \cdot \left(1 + \text{erf}\left(\frac{1}{\sqrt{2}} \cdot x\right)\right) + \frac{x}{\sqrt{2\pi}} \cdot e^{-\frac{x^2}{2}}$$

This is our final result:

$$\frac{df}{dx} = \frac{1}{2} \cdot \left(1 + \text{erf}\left(\frac{1}{\sqrt{2}} \cdot x\right)\right) + \frac{x}{\sqrt{2\pi}} \cdot e^{-\frac{x^2}{2}}$$


## Approximate Derivative

Starting with:

$$f(x) = 0.5 \cdot x \cdot (1 + \tanh(\sqrt{\frac{2}{\pi}} \cdot x \cdot (1 + 0.044715 \cdot x^2)))$$

For simplicity, let's denote:

$$z(x) = \sqrt{\frac{2}{\pi}} \cdot x \cdot (1 + 0.044715 \cdot x^2) = x \cdot (a + b \cdot x^2)$$

and 

$$v(x) = 1 + \tanh(z(x))$$

Then:

$$f(x) = 0.5 \cdot x \cdot (1 + \tanh(z(x))) = 0.5 \cdot x \cdot v(x)$$

Using the product rule: 

$$\frac{d}{dx}[u(x) \cdot v(x)] = u'(x) \cdot v(x) + u(x) \cdot v'(x)$$

Let:
- $u(x) = 0.5 \cdot x$
- $v(x) = 1 + \tanh(z(x))$

Step 1: Find $u'(x)$

$$u'(x) = 0.5$$

Step 2: Find $v'(x)$
Using the chain rule and the fact that the derivative of $\tanh(x)$ is $1 - \tanh^2(x)$:

$$v'(x) = (1 - \tanh^2(z(x))) \cdot z'(x)$$

The derivative of $z(x)$:

$$z'(x) = a + 3b \cdot x^2$$

Step 3: Using the identity for $1 - \tanh^2(z(x))$:

$$1 - \tanh^2(z(x)) = (1 - \tanh(z(x)))(1 + \tanh(z(x)))$$

$$1 - \tanh^2(z(x)) = (2 - (1 + \tanh(z(x))))(1 + \tanh(z(x)))$$

$$1 - \tanh^2(z(x)) = (2- v(x))v(x)$$

This confirms our identity. Now using the form with $(2 - (1 + \tanh(z(x))))$:

$$v'(x) = (2- v(x))v(x) \cdot z'(x)$$

Step 5: Apply the product rule for the complete derivative:

$$\frac{df}{dx} = 0.5 \cdot (1 + \tanh(z(x))) + 0.5 \cdot x \cdot (2- v(x))v(x) \cdot z'(x)$$

Substituting $z'(x) = a + 3b \cdot x^2$:

$$\frac{df}{dx} = 0.5 \cdot v(x) + 0.5 \cdot x \cdot (2 - v(x)) \cdot v(x) \cdot (a + 3b \cdot x^2)$$

$$\frac{df}{dx} = 0.5 \cdot v(x) \cdot \left[1 + x \cdot (2 - v(x)) \cdot (a + 3b \cdot x^2)\right]$$

