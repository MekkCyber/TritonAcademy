# Derivative of GeLU

## Exact Derivative

Starting with:
$$f = \frac{1}{2} \cdot e \cdot \left(1 + \text{erf}\left(\frac{1}{\sqrt{2}} \cdot e\right)\right)$$

Using the product rule: $\frac{d}{dx}[u(x) \cdot v(x)] = u'(x) \cdot v(x) + u(x) \cdot v'(x)$

Let:

- $u(e) = \frac{1}{2} \cdot e$
- $v(e) = 1 + \text{erf}\left(\frac{1}{\sqrt{2}} \cdot e\right)$

Step 1: Find $u'(e)$
$$u'(e) = \frac{d}{de}\left[\frac{1}{2} \cdot e\right] = \frac{1}{2}$$

Step 2: Find $v'(e)$
We need the chain rule here. The derivative of erf(x) is $\frac{2}{\sqrt{\pi}} \cdot e^{-x^2}$

$$v'(e) = \frac{d}{de}\left[1 + \text{erf}\left(\frac{1}{\sqrt{2}} \cdot e\right)\right] = \frac{d}{de}\left[\text{erf}\left(\frac{1}{\sqrt{2}} \cdot e\right)\right]$$

Using the chain rule with $g(e) = \frac{1}{\sqrt{2}} \cdot e$:
$$v'(e) = \frac{2}{\sqrt{\pi}} \cdot e^{-\left(\frac{1}{\sqrt{2}} \cdot e\right)^2} \cdot \frac{d}{de}\left[\frac{1}{\sqrt{2}} \cdot e\right]$$

$$v'(e) = \frac{2}{\sqrt{\pi}} \cdot e^{-\frac{e^2}{2}} \cdot \frac{1}{\sqrt{2}}$$

$$v'(e) = \frac{2}{\sqrt{\pi}} \cdot \frac{1}{\sqrt{2}} \cdot e^{-\frac{e^2}{2}}$$

$$v'(e) = \frac{2}{\sqrt{2\pi}} \cdot e^{-\frac{e^2}{2}}$$

Step 3: Apply the product rule
$$\frac{df}{de} = u'(e) \cdot v(e) + u(e) \cdot v'(e)$$

$$\frac{df}{de} = \frac{1}{2} \cdot \left(1 + \text{erf}\left(\frac{1}{\sqrt{2}} \cdot e\right)\right) + \frac{1}{2} \cdot e \cdot \frac{2}{\sqrt{2\pi}} \cdot e^{-\frac{e^2}{2}}$$

$$\frac{df}{de} = \frac{1}{2} \cdot \left(1 + \text{erf}\left(\frac{1}{\sqrt{2}} \cdot e\right)\right) + \frac{e}{\sqrt{2\pi}} \cdot e^{-\frac{e^2}{2}}$$

This is our final result:
$$\frac{df}{de} = \frac{1}{2} \cdot \left(1 + \text{erf}\left(\frac{1}{\sqrt{2}} \cdot e\right)\right) + \frac{e}{\sqrt{2\pi}} \cdot e^{-\frac{e^2}{2}}$$


## Approximate Derivative

Starting with:
$$f(e) = 0.5 \cdot e \cdot (1 + \tanh(\sqrt{\frac{2}{\pi}} \cdot e \cdot (1 + 0.044715 \cdot e^2)))$$

For simplicity, let's denote:
$$z(e) = \sqrt{\frac{2}{\pi}} \cdot e \cdot (1 + 0.044715 \cdot e^2) = e \cdot (a + b \cdot e^2)$$
and 
$$v(e) = 1 + \tanh(z(e))$$
Then:
$$f(e) = 0.5 \cdot e \cdot (1 + \tanh(z(e))) = 0.5 \cdot e \cdot v(e)$$

Using the product rule: $\frac{d}{dx}[u(x) \cdot v(x)] = u'(x) \cdot v(x) + u(x) \cdot v'(x)$

Let:
- $u(e) = 0.5 \cdot e$
- $v(e) = 1 + \tanh(z(e))$

Step 1: Find $u'(e)$
$$u'(e) = 0.5$$

Step 2: Find $v'(e)$
Using the chain rule and the fact that the derivative of $\tanh(x)$ is $1 - \tanh^2(x)$:
$$v'(e) = (1 - \tanh^2(z(e))) \cdot z'(e)$$

The derivative of $z(e)$:
$$z'(e) = a + 3b \cdot e^2$$

Step 3: Using the identity for $1 - \tanh^2(z(e))$:
$$1 - \tanh^2(z(e)) = (1 - \tanh(z(e)))(1 + \tanh(z(e)))$$
$$1 - \tanh^2(z(e)) = (2 - (1 + \tanh(z(e))))(1 + \tanh(z(e)))$$
$$1 - \tanh^2(z(e)) = (2- v(e))v(e)$$

This confirms our identity. Now using the form with $(2 - (1 + \tanh(z(e))))$:
$$v'(e) = (2- v(e))v(e) \cdot z'(e)$$

Step 5: Apply the product rule for the complete derivative:
$$\frac{df}{de} = 0.5 \cdot (1 + \tanh(z(e))) + 0.5 \cdot e \cdot (2- v(e))v(e) \cdot z'(e)$$

Substituting $z'(e) = a + 3b \cdot e^2$:
$$\frac{df}{de} = 0.5 \cdot v(e) + 0.5 \cdot e \cdot (2 - v(e)) \cdot v(e) \cdot (a + 3b \cdot e^2)$$

$$\frac{df}{de} = 0.5 \cdot v(e) \cdot \left[1 + e \cdot (2 - v(e)) \cdot (a + 3b \cdot e^2)\right]$$
