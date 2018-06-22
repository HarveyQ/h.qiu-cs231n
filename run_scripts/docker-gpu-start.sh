#!/bin/bash
do nvidia-docker run -i -t \
		-v ~/ssdUbu/courses/cs231n_qiuhuaqi:/home/cs231n \
		-p 8888:8888 \
		--rm=true \
		qiuhuaqi/cs231n-tf-gpu:cudnn7.0.5 \
		sh -c "jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --notebook-dir='/home/cs231n' "
