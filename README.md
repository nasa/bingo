![Bingo Logo](media/logo.png)

master: [![Build Status](https://github.com/nasa/bingo/actions/workflows/validation.yml/badge.svg?branch=main)](https://github.com/nasa/bingo/actions?query=branch%3Amain)

develop:
[![Build Status](https://github.com/nasa/bingo/actions/workflows/validation.yml/badge.svg?branch=develop)](https://github.com/nasa/bingo/actions?query=branch%3Adevelop)

## Description
Bingo is an open source package for performing symbolic regression, though it 
can be used as a general purpose evolutionary optimization package.  

## Key Features
*   Integrated local optimization strategies
*   Parallel island evolution strategy implemented with mpi4py
*   Coevolution of fitness predictors

# Quick Start

## Documentation
[Full Documentation Here](https://nasa.github.io/bingo/)

## Installation

```sh
pip install bingo-nasa
```

## Usage Example
A no-fuss way of using Bingo is by using the scikit-learn wrapper:
`SymbolicRegressor`. Let's setup a test case to show how it works.

### Setting Up the Regressor

Configure `SymbolicRegressor` with its population size, minimum and maximum
expression stack sizes, and simplification mode. The estimator uses
Expression-native generation, fitting, and loss evaluation. See the
[migration guide](docs/source/migration.rst) for the current public API.
Evolution minimizes loss, while Expression scores are higher-is-better. Legacy
AGraph/Equation checkpoints are not compatible with the Expression API.


```python
from bingo.symbolic_regression import SymbolicRegressor
regressor = SymbolicRegressor(
    population_size=100,
    min_stack_size=8,
    max_stack_size=16,
    simplification="cas",
)
```

### Training Data
Here we're just creating some dummy training data from the equation $5.0 X_0^2 + 3.5 X_0$. More on training data can be found
in the [data formatting guide](https://nasa.github.io/bingo/_high_level/data_formatting.html).
```python
import numpy as np
X_0 = np.linspace(-10, 10, num=30).reshape((-1, 1))
X = np.array(X_0)
y = (5.0 * X_0 ** 2 + 3.5 * X_0).ravel()
```


```python
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.xlabel("X_0")
plt.ylabel("y")
plt.title("Training Data")
plt.show()
```


    
![png](media/usage_example_1.png)
    


### Fitting the Regressor

Fitting is as simple as calling the `.fit()` method.


```python
regressor.fit(X, y)
```


### Getting the Best Individual


```python
best_individual = regressor.get_best_individual()
print("best individual is:", best_individual)
```

    best individual is: X_0 + (5.0)((0.4999999999999999)(X_0) + (X_0)(X_0))


### Predicting Data with the Best Individual

You can use the regressor's `.predict(X)` or the best individual's underlying
expression's `.predict(X)` to get predictions for `X`.


```python
pred_y = regressor.predict(X)
pred_y = best_individual.expression.predict(X)

plt.scatter(X, y)
plt.plot(X, pred_y, 'r')
plt.xlabel("X_0")
plt.ylabel("y")
plt.legend(["Actual", "Predicted"])
plt.show()
```


    
![png](media/usage_example_2.png)

# Source

## Installation from Source

For those looking to develop their own features in Bingo.

First clone the repo and move into the directory:

```sh
git clone https://github.com/nasa/bingo.git
cd bingo
```

Then make sure you have the requirements necessary to use Bingo:


```sh
conda env create -f conda_environment.yml
```

or

```sh
pip install -r requirements.txt
```

(Optional) Build the C++ expression backend:

```sh
./.build_cppagraph.sh
```

Now you should be good to go! You can run Bingo's test suite to make sure that
the installation process worked properly:

```sh
pytest tests
```

Add Bingo to your Python path to begin using it from other directories.

```sh
export PYTHONPATH="$PYTHONPATH:/path/to/bingo/"
```

and test it with:

```sh
python -c 'import bingo; from bingo.expressions import AGraphExpression'
```

## Contributing
1.  Fork it (<https://github.com/nasa/bingo/fork>)
2.  Create your feature branch (`git checkout -b feature/fooBar`)
3.  Commit your changes (`git commit -am 'Add some fooBar'`)
4.  Push to the branch (`git push origin feature/fooBar`)
5.  Create a new Pull Request

# Citing Bingo
Please consider citing the following reference when using bingo in your works.

### MLA:
Randall, David L., et al. "Bingo: a customizable framework for symbolic regression with genetic programming." Proceedings of the Genetic and Evolutionary Computation Conference Companion. 2022.

### Bibtex:
```latex
@inproceedings{randall2022bingo,
  title={Bingo: a customizable framework for symbolic regression with genetic programming},
  author={Randall, David L and Townsend, Tyler S and Hochhalter, Jacob D and Bomarito, Geoffrey F},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference Companion},
  pages={2282--2288},
  year={2022}
}
```

# Versioning
We use [SemVer](http://semver.org/) for versioning. For the versions available, 
see the [tags on this repository](https://github.com/nasa/bingo/tags). 

# Authors
*   Geoffrey Bomarito
*   Tyler Townsend
*   Jacob Hochhalter
*   David Randall
*   Ethan Adams
*   Kathryn Esham
*   Diana Vera
  
# License 
Copyright 2018 United States Government as represented by the Administrator of 
the National Aeronautics and Space Administration. No copyright is claimed in 
the United States under Title 17, U.S. Code. All Other Rights Reserved.

The Bingo Mini-app framework is licensed under the Apache License, Version 2.0 
(the "License"); you may not use this application except in compliance with the 
License. You may obtain a copy of the License at 
http://www.apache.org/licenses/LICENSE-2.0 .

Unless required by applicable law or agreed to in writing, software distributed 
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied. See the License for the 
specific language governing permissions and limitations under the License.
