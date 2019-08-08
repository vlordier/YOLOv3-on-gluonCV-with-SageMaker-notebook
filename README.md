# Transfer Learning YOLOv3 on GluonCV using SageMaker notebook and exporting as CoreML

This is a boilerplate to retrain [YOLO v3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) & [see Medium post](https://medium.com/diaryofawannapreneur/yolo-you-only-look-once-for-object-detection-explained-6f80ea7aaa1e) [object detection](https://en.wikipedia.org/wiki/Object_detection) on [Gluon CV](https://gluon-cv.mxnet.io/build/examples_detection/train_yolo_v3.html), based on [Apache MXNnet](https://mxnet.apache.org/) using Amazon AWS SageMaker Notebook Instances (https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html) for Apple iOS [CoreML](https://www.youtube.com/watch?v=T4t73CXB7CU).
It returns the newly trained model which can be used on iOS with CoreML.

### DarkNet Yolov3 > ONNX > GluonCV > ONNX > CoreML

We fine-tune an existing trained model on our new categories, using Transfer Learning. [12]
First, we get Yolo weights trained on [COCO dataset](http://cocodataset.org/) then [convert to ONNX](https://mxnet.incubator.apache.org/versions/master/tutorials/onnx/super_resolution.html) to import into GluonCV.
Then we retrain using MXNET on AWS GPU instance.
Then we export and convert to CoreML format, ready to be used.


#### What it does : 
1. takes your data on S3
2. [augments your data](https://gluon-cv.mxnet.io/api/data.transforms.html#gluoncv.data.transforms.presets.yolo.YOLO3DefaultTrainTransform)
2. trains YOLO v3 from a pretrained model ([transfer learning](https://gluon-cv.mxnet.io/build/examples_detection/finetune_detection.html))
3. deploys a SageMaker endpoint for images or video stream, to test it out with the this webapp [13]
4. shows nice metrics to evaluate your model
5. saves your GluonCV model arteficts to s3
6. [exports MXNET model artifacts to ONNX](https://github.com/onnx/tutorials/blob/master/tutorials/MXNetONNXExport.ipynb)
7. [converts ONNX to CoreML](https://github.com/onnx/onnx-coreml)
8. 

#### To use
0. make sure you have access to GPUs in your AWS region, or request one
1. [clone](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-notebooks-now-support-git-integration-for-increased-persistence-collaboration-and-reproducibility/) this repo into your Notebook Instance when creating or updating the notebook 
2. [upload](https://docs.aws.amazon.com/AmazonS3/latest/user-guide/upload-objects.html) your data to AWS S3 
3. update the locations for S3 buckets
4. the rest is in the notebook and pretty straightforward





