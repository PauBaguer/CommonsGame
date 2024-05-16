FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

LABEL authors="paubaguer"

RUN apt update
RUN apt install -y git
# Install requirements
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt
ENTRYPOINT ["top", "-b"]