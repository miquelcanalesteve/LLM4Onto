# LLM4Onto


Primer docker:

docker build -t data_prep -f docker/Dockerfile.data_prep .
docker run --rm -it -v $(pwd)/src/data_prep:/app data_prep


