FROM python:3

# metadata
LABEL maintainer="Kasper D. Fischer <kasper.fischer@rub.de>"
LABEL version="0.2-docker"
LABEL description="Docker image for the survBot application"

# install required system packages
RUN apt update && apt install -y bind9-host iputils-ping

# create user and group and home directory
RUN groupadd -r survBot && useradd -r -g survBot survBot
RUN mkdir -p /home/survBot && chown -R survBot:survBot /home/survBot

# change working directory
WORKDIR /usr/src/app

# install required python packages
RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    pip install --no-cache-dir --requirement /tmp/requirements.txt

# copy application files
COPY survBot.py utils.py write_utils.py LICENSE README.md ./

# copy configuration files
VOLUME /usr/src/app/conf
COPY parameters.yaml mailing_list.yam[l] simulate_fail.jso[n] conf/
RUN ln -s conf/simulate_fail.json simulate_fail.json

# copy www files
VOLUME /usr/src/app/www
COPY logo.pn[g] stylesheets/desktop.css stylesheets/mobile.css www/
RUN ln -s www/survBot_out.html www/index.html

# change ownership of working directory
RUN chown -R survBot:survBot /usr/src/app

# run the application as user survBot
USER survBot
CMD [ "python", "./survBot.py", "-html", "www", "-parfile", "conf/parameters.yaml" ]
