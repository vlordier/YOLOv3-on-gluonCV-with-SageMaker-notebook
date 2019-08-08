# YOLOv3-on-gluonCV-with-SageMaker-notebook

This is a boilerplate to retrain a CoreML YOLO v3 [1] & [2] object detection [8] on Gluon CV [4], based on Apache MXNet ([3]) using Amazon AWS SageMaker Notebook Instances [5] 
It returns the newly trained model which can be used on iOS with CoreML

### DarkNet Yolov3 > ONNX > GluonCV > ONNX > CoreML

We fine-tune an existing trained model on our new categories, using Transfer Learning. [12]
First, we get Yolo weights trained on COCO [15] then convert to ONNX [14] to import into GluonCV.
Then we retrain using MXNET on AWS GPU instance.
Then we export and convert to CoreML format, ready to be used.


#### What it does : 
1. takes your data on S3
2. augments your data [11]
2. trains YOLO v3 from a pretrained model (transfer learning [12])
3. deploys a SageMaker endpoint for images or video stream, to test it out with the this webapp [13]
4. shows nice metrics to evaluate your model
5. saves your GluonCV model arteficts to s3
6. exports MXNET model artifacts to ONNX [10]
7. converts ONNX to CoreML [9]
8. 

#### To use
0. make sure you have access to GPUs in your AWS region, or request one
1. clone into your Notebook Instance when creating or updating the notebook https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-notebooks-now-support-git-integration-for-increased-persistence-collaboration-and-reproducibility/
2. upload your data to AWS S3 https://docs.aws.amazon.com/AmazonS3/latest/user-guide/upload-objects.html
3. update the locations for S3 buckets
4. the rest is in the notebook and pretty straightforward


[1] YOLOv3 paper https://pjreddie.com/media/files/papers/YOLOv3.pdf
[2] YOLOv3 Medium post explanation https://medium.com/diaryofawannapreneur/yolo-you-only-look-once-for-object-detection-explained-6f80ea7aaa1e
[3] MXNET https://mxnet.apache.org/
[4] Gluon CV tutorial https://gluon-cv.mxnet.io/build/examples_detection/train_yolo_v3.html
[5] AWS Notebook SageMaker Instances https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html
[6] Siraj Raval explaination on how to use CoreML https://www.youtube.com/watch?v=T4t73CXB7CU
[7] Netron vizualiser https://github.com/lutzroeder/netron
[8] https://en.wikipedia.org/wiki/Object_detection
[9] https://github.com/onnx/onnx-coreml
[10] https://github.com/onnx/tutorials/blob/master/tutorials/MXNetONNXExport.ipynb
[11] https://gluon-cv.mxnet.io/api/data.transforms.html#gluoncv.data.transforms.presets.yolo.YOLO3DefaultTrainTransform
[12] https://gluon-cv.mxnet.io/build/examples_detection/finetune_detection.html
[13] coming soon
[14] https://mxnet.incubator.apache.org/versions/master/tutorials/onnx/super_resolution.html
[15] http://cocodataset.org/



