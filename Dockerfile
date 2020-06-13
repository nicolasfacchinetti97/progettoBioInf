FROM tensorflow/tensorflow:2.1.0-py3

EXPOSE 5000
WORKDIR /progettobioinf

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "progettobioinf/entry_point.py" ]