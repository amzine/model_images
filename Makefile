IMAGE_NAME=disease-classifier
CONTAINER_NAME=disease-classifier
build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -d --name $(CONTAINER_NAME) -p 5002:5002 $(IMAGE_NAME)
stop:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)

clean:
	docker rmi $(IMAGE_NAME) -f

all: build run

