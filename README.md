
# A2e2g

Integrated code for determining market participation of wind energy. 
A2e2g provides a framework to enable wind plants to provide grid services, 
including:
* Day-ahead atmospheric forecasting
* Probablisitc power forecasting (day-ahead)
* Day-ahead market offering
* Day-ahead market simulation
* Short-term atmospheric and power forecasting (for real-time market)
* Real-time offering
* Real-time market simulation and signal generation
* Wind plant real-time control
* Wind plant aerodynamic simlations

Please refer to the [A2e2g project report](https://www.osti.gov/biblio/1962807) 
for a full 
description of the project. 

## Installation

The A2e2g code can be installed by downloading the git repository
from GitHub with ``git`` and using ``pip`` to locally install it.
It is highly recommended to use a [conda](https://docs.conda.io/en/latest/miniconda.html) virtual environment. The following
commands in a terminal or shell can be used to download and install A2e2g.

```bash
    # Download the source code from github
    git clone https://github.com/NREL/a2e2g

    # Create a conda environment from the included yml
    cd a2e2g
    conda env create --name a2e2g-env --file=environment.yml
    conda activate a2e2g-env
    
    # Install A2e2g
    pip install -e .
```


## Getting started

An example that demonstrates the use of the A2e2g platform and its modules 
is provided in example_script.py. To run, execute the following commands.
```bash
    # Navigate to the a2e2g main directory, if not there already

    # Run the python example
    python example_script.py
```
The data needed to run the example is not included in the repository for
data storage reasons. Please contact us if you need this data.

We also provide an example Jupyter notebook (notebook_example.ipynb) 
that runs through the same code as main.py step by step, including generating
some intermediate plots. This should be run 
using a2e2g-env as the kernel.

## A2e2g code structure

The main source code form the A2e2g platform is found under a2e2g > modules.
This contains a subdirectory pertaining to each important function of the 
platform. The __forecast__, __power_forecast__, and __market__ submodules 
contain the models that produce the weather forecasts, power forecasts and market 
signals, respectively. The __control__ submodule contains the various plant-level 
real-time aerodynamic control strategies for following power setpoints from the system 
operator. Finally, the __control_simulation__ submodule provides tools for testing the 
aerodynamic controllers interacting with a simulated wind plant. For a description of 
each module,
please refer to the 
[A2e2g project report](https://www.osti.gov/biblio/1962807).

The main driver is a2e2g > a2e2g.py, which contains most calls to each of the 
modules.

Users will also find dummy data for example simulations in a2e2g > data, and 
various other scripts that play a smaller role in the platform under 
a2e2g > supporting_scripts.


## License

Copyright 2023 Alliance for Sustainable Energy, LLC
 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 
http://www.apache.org/licenses/LICENSE-2.0
 
unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.