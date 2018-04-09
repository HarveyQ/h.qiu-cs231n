#!/bin/bash
docker run -i -t \
         -v /Users/HarveyQ/Documents/mlcv/deep_learning/cs231n/cs231n-2017:/home/cs231n \
	        -p 8888:8888 \
          --rm=true \
	        qiuhuaqi/cs231n-tf-cpu \
	        sh -c "jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root --notebook-dir='/home/cs231n'"
