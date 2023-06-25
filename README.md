# TMM numerical package, and Forward + Reverse ML models

## Overview

This repository contains two packages and a collection of jupyter notebooks that were used to design and train the forward and reverse Machine Learning models to be used in the L2 Nano-electronics practical course.

## Prerequisites

This project requires python at version 3.10.

## Installation

This repository contains two packages which must be installed in your python environment. The easiest way to do this, which does not involve publishing them to a package registry is to install them as editable packages. `Poetry` is a relatively new python package manager, and makes this incredibly easy. Installation instructions for poetry can be found [here](https://python-poetry.org/docs/). If it is not possible to install poetry, is is possible to setup this repository to work with just a standard python installation.

### Option 1: Poetry

If poetry is installed then simply install the repository and its dependencies in a poetry environment using

```bash
poetry install
```

Then activate that environment using

```bash
poetry shell
```

### Option 2: pip

included as extra to `pyproject.toml` in this repository is a `requirements.txt` file this is an old package management format that can be used by the python built-in package manager `pip`. To install the dependencies listed in this file:

First create a virtual python environment to ensure that any dependencies do not conflict with dependencies already installed in the global environment.
```bash
python -m venv ./.venv
```

Then activate the environment:

```bash
./.venv/bin/activate
```

(If you are on windows, the command looks a bit different):

```bash
./.venv/scripts/activate.ps1
```

With the virtual environment activated the dependencies can be installed:

```bash
pip install -r ./requirements.txt
```

---
## Running the notebooks

The notebooks can be run using jupyter, if you are familiar with jupyter notebooks then they can be run using your preferred method, but if not, you can interact with them easily using the web interface by running the following command with the virtual environment activated.

```bash
jupyter notebook
```