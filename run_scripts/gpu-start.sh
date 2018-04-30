#!/bin/bash
sudo nvidia-docker run -i -t \
		-v ~/ssdUbu/cs_courses/cs231n_qiuhuaqi:/home/cs231n \
		-p 8888:8888 \
		--rm=true \
		qiuhuaqi/cs231n-tf-gpu \
		sh -c "jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --notebook-dir='/home/cs231n' "
