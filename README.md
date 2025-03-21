# survBot

version: 0.2

survBot is a small program used to track station quality channels of DSEBRA stations via PowBox output over SOH channels
 by analyzing contents of a Seiscomp data archive.

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

The main program with html output is executed by entering

```shell script
python survBot.py -html path_for_html_output
```

There are example stylesheets in the folder *stylesheets* that can be copied into the path_for_html_output if desired.

The GUI can be loaded via

```shell script
python survBotGui.py
```

### Docker

To run the program in a Docker container, first build the image:

```shell script
docker build -t survbot .
```

Then run the container:

```shell script
docker run -v /path/to/conf-dir:/usr/src/app/conf -v /path/to/output:/usr/src/app/www survbot
```

The directory `/path/to/conf-dir` should contain the `parameters.yaml` file, and the directory `/path/to/output` will contain the output HTML files.

### Configuration of the e-mail server settings

The e-mail server settings can be configured in the `parameters.yaml` file. The following settings are available:

* `mailserver`: the address of the mail server
* `auth_type`: the authentication type for the mail server (`None`, `SSL`, `TLS`)
* `port`: the port of the mail server
* `user`: the username for the mail server (if required)
* `password`: the password for the mail server (if required)

The `user` and `password` fields are optional, and can be left empty if the mail server does not require authentication. The `auth_type` field can be set to `None` if no authentication is required, `SSL` if the mail server requires SSL authentication, or `TLS` if the mail server requires TLS authentication. If the `user` or `password` fields are set to `Docker` ore `ENV` the program will try to read the values from the docker secrets `mail_user` and `mail_password` or environment variables `MAIL_USER` and `MAIL_PASSWORD` respectively. Docker secrets are only available in Docker Swarm mode, i.e. if the program is run as a service.

## Version Changes

### 0.2

* surveillance of mass, clock and gaps
* individual mailing lists for different stations
* html mail with recent status information
* updated web page design
* restructured parameter file
* recognize if PBox is disconnected

### 0.2-docker

* added Dockerfile for easy deployment
* added more settings for connection to a mail server

## Staff

Original author: M.Paffrath (<marcel.paffrath@rub.de>)
Contributions: Kasper D. Fischer (<kasper.fischer@rub.de>)

Jan 2025
