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

## Usage Example
Currently, the best place to get started in bingo is by going through some 
examples.  The examples directory contains several python scripts and jupyter 
notebook examples of varying complexity.  The simplegp examples illustrate the 
very basics of symbolic regression in bingo, utilizing a simplified interface.  
More complex (and more efficient) examples can be found as well.

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
