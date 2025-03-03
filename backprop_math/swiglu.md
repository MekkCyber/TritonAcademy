# Derivative of SwiGLU


we have :
$$f(x) = \frac{x}{1 + e^{-x}}$$

Find $\frac{df}{dx}$ using the quotient rule
For $f(x) = \frac{u(x)}{v(x)}$, the quotient rule gives us:
$$\frac{df}{dx} = \frac{u'(x) \cdot v(x) - u(x) \cdot v'(x)}{v(x)^2}$$

Where:
- $u(x) = x$
- $v(x) = 1 + e^{-x}$

Calculate $u'(x)$
$$u'(x) = \frac{d}{dx}[x] = 1$$

Calculate $v'(x)$
$$v'(x) = \frac{d}{dx}[1 + e^{-x}] = -e^{-x}$$

We apply the quotient rule
$$\frac{df}{dx} = \frac{1 \cdot (1 + e^{-x}) + x \cdot e^{-x}}{(1 + e^{-x})^2}$$

$$\frac{df}{dx} = \frac{1 + e^{-x} + x \cdot e^{-x}}{(1 + e^{-x})^2}$$

$$\frac{df}{dx} = \frac{1}{1 + e^{-x}} + \frac{x \cdot (e^{-x} + 1)}{(1 + e^{-x})^2} - \frac{x}{(1 + e^{-x})^2}$$

Alternative expression using sigmoid function
Since $s = \sigma(x) = \frac{1}{1 + e^{-x}}$, we can write:

$$\frac{df}{dx} = \sigma(x) + x \cdot \sigma(x) - x \cdot \sigma(x)^2 = \sigma(x) \cdot (1 + x \cdot (1 - \sigma(x)))$$
