run:
	docker build -t yolov3 .

in:
	NV_GPU=2 nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app/sig --rm -it yolov3:latest bash
