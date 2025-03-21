FROM python:3

# metadata
LABEL maintainer="Kasper D. Fischer <kasper.fischer@rub.de>"
LABEL version="0.2-docker"
LABEL description="Docker image for the survBot application"

# install required system packages
RUN apt update && apt install -y bind9-host iputils-ping

# create user and group
RUN groupadd -r survBot && useradd -r -g survBot survBot

# change working directory
WORKDIR /usr/src/app

# install required python packages
RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    pip install --no-cache-dir --requirement /tmp/requirements.txt

# switch to user survBot
USER survBot

# copy application files
COPY survBot.py utils.py write_utils.py LICENSE README.md ./

# copy configuration files
VOLUME /usr/src/app/conf
COPY parameters.yaml mailing_list.yaml simulate_fail.json conf/
RUN ln -s conf/simulate_fail.json simulate_fail.json

# copy www files
VOLUME /usr/src/app/www
COPY logo.png stylesheets/desktop.css stylesheets/mobile.css www/
RUN ln -s www/index.html survBot_out.html

# run the application
CMD [ "python", "./survBot.py", "-html", "www", "-parfile", "conf/parameters.yaml" ]
