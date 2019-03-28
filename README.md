![Bingo Logo](media/logo.png)

master: [![Build Status](https://travis-ci.com/nasa/bingo.svg?branch=master)](https://travis-ci.com/nasa/bingo) [![Coverage Status](https://coveralls.io/repos/github/nasa/bingo/badge.svg?branch=master)](https://coveralls.io/github/nasa/bingo?branch=master)

develop: [![Build Status](https://travis-ci.com/nasa/bingo.svg?branch=develop)](https://travis-ci.com/nasa/bingo) [![Coverage Status](https://coveralls.io/repos/github/nasa/bingo/badge.svg?branch=develop)](https://coveralls.io/github/nasa/bingo?branch=develop) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/9fe09cffafe64032962a82f7f1588e9f)](https://www.codacy.com/app/bingo_developers/bingo?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=nasa/bingo&amp;utm_campaign=Badge_Grade)

## General
Bingo is an open source package for performing symbolic regression, Though it 
can be used as a general purpose evolutionary optimization package.  

### Key Features
  * Integrated local optimization strategies
  * Parallel island evolution strategy implemented with mpi4py
  * Coevolution of fitness predictors
  
### Note
At this point the develop branch is significantly ahead of the master 
branch but will not be released officially until a larger proportion of the 
current master features are supported.

## Getting Started

### Dependencies
Bingo is intended for use with Python 3.x.  Bingo requires installation of a 
few dependencies which are relatively common for data science work in python:
  - numpy
  - scipy
  - matplotlib
  - mpi4py (if parallel implementations are to be run)
  - pytest (if the testing suite is to be run)
  
A requirements.txt file is included for easy installation of dependecies with 
pip or conda.

Installation with pip:
```
pip install -r requirements.txt
```

Installation with conda:
```
conda install --yes --file requirements.txt
```

### BingoCpp
A section of bingo is written in c++ for increased performance.  In order to 
take advantage of this capability, the code must be compiled.  See the 
documentation in the bingocpp submodule for more information.

Note that bingo can be run without the bingocpp portion, it will just have lower 
performance.

If bingocpp has been properly installed, the following command should run 
without error.
```
python -c "import bingocpp"
```

A common error in the installation of bingocpp is that it must be built with 
the same version of python that will run your bingo scripts.  The easiest way 
to ensure consistent python versioning is to build and run in a Python 3 
virtual environment.

### Documentation
Sphynx is used for automatically generating API documentation for bingo. The 
most recent build of the documentation can be found in the repository at: 
doc/_build/html/index.html 


## Running Tests
An extensive unit test suite is included with bingo to help ensure proper 
installation. The tests can be run using pytest on the tests directory, e.g., 
by running:
``` 
pytest tests 
```
from the root directory of the repository.

## Usage Examples
In addition to the example sshown here, the best place to get started in bingo 
is by going through the [examples directory](examples/)5. It contains several scripts and 
jupyter notebooks.

### A simple evolutionary analysis with bingo: the one max problem
This example walks through the general steps needed to set up and run a bingo 
analysis.  The example problem described here is the one max problem. In the 
one max problem individuals in a population are defined by a chromosome with a 
list of 0 or 1 values, e.g., `[0, 1, 1, 0, 1]`.  The goal of the optimization 
is to evolve toward an optimum list containing all 1's.  A complete version of 
this example is script is found [here](examples/OneMaxExample.py).

#### Defining a chromosome generator
Bingo's built-in `MultipleValueChromosome` is used here.  Individuals of this 
contain their genetic information in a list attribute named `values`.  A 
chromosome generator is used to generate members of the population.  The 
`MultipleValueChromosomeGenerator` generates these individuals by populating 
the indivudual's `values` from a given input function.
```python 
import numpy as np
from bingo.MultipleValues import MultipleValueChromosomeGenerator
np.random.seed(0)  # seeded for reproducible results

def generate_0_or_1():
    return np.random.choice([0, 1])

generator = MultipleValueChromosomeGenerator(generate_0_or_1,
                                             values_per_chromosome=16) 
``` 

#### Defining the evolutionary algorithm
Evolutionary algorithms have 3 phases in bingo: variation, evaluation and 
selection.  The variation phase is responsible for generating offspring of the 
population, usually through some combination of mutation and crossover.  In 
this example `VarOr` is used which creates offspring through either mutation or 
crossover (never both).
```python
from bingo.MultipleValues import SinglePointCrossover, SinglePointMutation
from bingo.EA.VarOr import VarOr

crossover = SinglePointCrossover()
mutation = SinglePointMutation(generate_0_or_1)
variation_phase = VarOr(crossover, mutation,
                        crossover_probability=0.4,
                        mutation_probability=0.4)
```
The evaluation phase is responsible for evaluating the fitness of new members 
of a population.  It relies on the definition of a `FitnessFunction` class.  
The goal of bingo analyses is to *minimize* fitness, so fitness functions 
should be constructed accordingly.  In the one max problem fitness is defined 
as the number of 0's in the individuals `values`.
```python
from bingo.Base.FitnessFunction import FitnessFunction
from bingo.Base.Evaluation import Evaluation

class OneMaxFitnessFunction(FitnessFunction):
    """Callable class to calculate fitness"""
    def __call__(self, individual):
        return individual.values.count(0)

fitness = OneMaxFitnessFunction()
evaluation_phase = Evaluation(fitness)
```
The selection phase is responsible for choosing which members of the population 
proceed to the next generation.  An implementation of the common tournament 
selection algorithm is used here.
```python
from bingo.EA.TournamentSelection import Tournament

selection_phase = Tournament(tournament_size=2)
```
Based on these phases, an `EvolutionaryAlgorithm` can be made.
```python
from bingo.Base.EvolutionaryAlgorithm import EvolutionaryAlgorithm

ev_alg = EvolutionaryAlgorithm(variation_phase, evaluation_phase, 
                               selection_phase)
```

#### Creating an island and running the analysis
An `Island` is the fundamental unit in bingo evolutionary analyses.  It is 
responsible for generating and evolving a population (using a generator and 
evolutionary algorithm).
```python
from bingo.Island import Island

island = Island(ev_alg, generator, population_size=10)
best_individual = island.best_individual()
print("Best individual at start: ", best_individual)
print("Best individual's fitness: ", best_individual.fitness)
```
```
>> Best individual at start:  [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1]
>> Best individual's fitness:  5
```
The island can be evolved directly using it's `execute_generational_step` 
member function.  In this case the population is evolved for 50 generations
```python
for _ in range(50):
    island.execute_generational_step()

best_individual = island.best_individual()
print("Best individual at end: ", best_individual)
print("Best individual's fitness: ", best_individual.fitness)
```
```
>> Best individual at end:  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
>> Best individual's fitness:  0
```

## Contributing
1. Fork it (<https://github.com/nasa/bingo/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request


## Versioning
We use [SemVer](http://semver.org/) for versioning. For the versions available, 
see the [tags on this repository](https://github.com/nasa/bingo/tags). 

## Authors
  * Geoffrey Bomarito
  * Kathryn Esham
  * Ethan Adams
  * Tyler Townsend
  * Diana Vera
  
## Licence 
#### Notices
Copyright 2018 United States Government as represented by the Administrator of 
the National Aeronautics and Space Administration. No copyright is claimed in 
the United States under Title 17, U.S. Code. All Other Rights Reserved.
 
#### Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF 
ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED 
TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY 
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR 
FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR 
FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE 
SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN 
ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, 
RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS 
RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY 
DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF 
PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS." 
 
Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE 
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY 
PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY 
LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, 
INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE 
OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED 
STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR 
RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY 
SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
