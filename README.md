# survBot

version: 0.1

survBot is a small program used to track station quality channels of DSEBRA stations via PowBox output over SOH channels
 by analysing contents of a Seiscomp3 datapath.

## Requirements

The following packages are required:

* Python 3
* obspy
* pyyaml

(the following are dependencies of the above):

* numpy
* matplotlib

to use the GUI:

* PySide2, PyQt4 or PyQt5

## Usage

Configurations of *datapath*, *networks*, *stations* etc. can be done in the **parameters.yaml** input file.

The main program is executed by entering

```shell script
python survBot.py
```

The GUI can be loaded via

```shell script
python survBotGui.py
```

## Staff

Original author: M.Paffrath (marcel.paffrath@rub.de)

November 2022
