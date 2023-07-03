# run_EVA_generic
code to use pyextremes to run extreme value analysis with generic data and create publication quality plots

**License:** CC0-1.0

**Support:** please [create an issue](https://github.com/arfogg/run_EVA_generic/issues) or contact [arfogg](https://github.com/arfogg) directly. Any input on the code / issues found are greatly appreciated and will help to improve the software.

## Required Packages

matplotlib, numpy, pandas and pyextremes

[Install pyextremes following instructions from its github here](https://github.com/georgebv/pyextremes)


## Running the code

First, the code must be downloaded using `git clone https://github.com/arfogg/run_EVA_generic`

Then, from a python terminal:
`import run_extreme_value_analysis`

`run_extreme_value_analysis.run_eva(df, 'tag')`

Note that your input df must have datetime as an index - this is a requirement of pyextremes.



## Acknowledgements

ARF gratefully acknowledges the support Irish Research Council Government of Ireland Postdoctoral Fellowship GOIPD/2022/782.
