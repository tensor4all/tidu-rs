# Higher-Order AD

Higher-order derivatives are built by applying graph transforms repeatedly.

For `f: R^n -> R^m`, each `linearize` call contracts one tangent vector with the
next derivative tensor. The output shape remains `R^m`.

```text
1st order: J * dx1
2nd order: d(J * dx1) * dx2
3rd order: d(d(J * dx1) * dx2) * dx3
```

Use fresh `DiffPassId` values for repeated `linearize` calls. Resolve the graph
collection that includes prior transform results before transforming again.

Reverse-over-forward and forward-over-reverse pipelines combine `linearize` with
`linear_transpose`, then linearize the resulting graph again.
