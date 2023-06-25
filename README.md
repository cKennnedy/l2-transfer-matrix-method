# TMM numerical package, and Forward + Reverse ML models

## Overview

This repository contains two packages and a collection of jupyter notebooks that were used to design and train the forward and reverse Machine Learning models to be used in the L2 Nano-electronics practical course.

It also contains a number of saved keras models which are pre trained on the training data included in the repository. the versions marked FINAL are the ones used in the report.

## Prerequisites

This project requires python at version 3.10.

## Installation

This repository contains two packages which must be installed in your python environment. The easiest way to do this, which does not involve publishing them to a package registry is to install them as editable packages. `Poetry` is a relatively new python package manager, and makes this incredibly easy. Installation instructions for poetry can be found [here](https://python-poetry.org/docs/).

If poetry is installed then simply install the repository and its dependencies in a poetry environment using

```bash
poetry install
```

Then activate that environment using

```bash
poetry shell
```

---
## Running the notebooks

The notebooks can be run using jupyter, if you are familiar with jupyter notebooks then they can be run using your preferred method, but if not, you can interact with them easily using the web interface by running the following command with the virtual environment activated.

```bash
jupyter notebook
```