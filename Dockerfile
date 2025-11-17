FROM dustynv/l4t-pytorch:r36.4.0

ENV PIP_NO_CACHE_DIR="true"

RUN pip install --index-url=https://pypi.org/simple pip-mark-installed && \
    pip-mark-installed opencv-python # it *is* installed, but pip doesn't get the memo

RUN pip install --index-url=https://pypi.org/simple ultralytics mjpeg-streamer lapx
RUN pip install --index-url=https://wpilib.jfrog.io/artifactory/api/pypi/wpilib-python-release-2024/simple pyntcore

COPY . /root/repo

EXPOSE 5090/tcp
WORKDIR /root/repo
CMD "python3" "/root/repo/detect.py"

