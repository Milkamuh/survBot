FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p www
RUN ln -s www/survBot_out.html www/index.html
RUN cp stylesheets/*.css www/
RUN touch logo.png
RUN cp logo.png www/logo.png

CMD [ "python", "./survBot.py", "-html", "www" ]
