# YOLOv3-on-gluonCV-with-SageMaker-notebook

This is a boilerplate to train YOLO v3 object detection on Gluon CV (https://gluon-cv.mxnet.io/), based on Apache MXNet (https://mxnet.apache.org/) using Amazon AWS SageMaker Notebook Instances
(https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html)

To use, 
0- make sure you have access to GPUs in your AWS region, or request one
1- clone into your Notebook Instance when creating or updating the notebook https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-notebooks-now-support-git-integration-for-increased-persistence-collaboration-and-reproducibility/
2- upload your data to AWS S3 https://docs.aws.amazon.com/AmazonS3/latest/user-guide/upload-objects.html
3- update the locations for S3 buckets
4- the rest is in the notebook and pretty straightforward
