# v0.4

## v0.4.2

* `SelectDim` is no longer type unstable -- Internal storage for the Layer has been changed
* `Dropout` & `VariationalDropout` return `NoOpLayer` if the probability of dropout is `0`
* Code Formatting -- SciMLStyle (https://github.com/avik-pal/Lux.jl/pull/31)

## v0.4.1

* Fix math rendering in docs
* Add Setfield compat for v1.0