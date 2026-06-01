# Complex AD

`tidu` follows the JAX convention for complex automatic differentiation.

Forward mode computes the full real-linear derivative:

```text
df = (df/dz) * dz + (df/dconj(z)) * conj(dz)
```

Reverse mode transposes linear maps with respect to the real inner product:

```text
<a, b> = Re(conj(a) * b)
```

For a general function `f: C -> C`, a cotangent is:

```text
ct_z = ct_y * conj(df/dz) + conj(ct_y) * (df/dconj(z))
```

For real losses, this gives `ct_z = 2 * dL/dconj(z)` when the output cotangent
seed is `1`. That differs from frameworks that directly return `dL/dconj(z)`,
but the steepest-descent direction is the same.
