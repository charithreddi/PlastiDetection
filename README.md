# PlastiDetection

An award winning project 

A mobile app that can detect plastic bottles from real time streaming



This project has two main goals:
-	Plastic bottle detection
-	Potential use case for removing detected bottles.
  
The main purpose of this project is to be able to train and deploy a model using a mobile app so that it can detect plastic bottles from real time streaming. Furthermore, we can create an alert for any plastic bottles found with the goal of notifying a user so they can easily remove that bottle from their environment. The project attempts to use existing publicly available Kaggle dataset to detect plastic bottles; the dataset has 8000 annotated images and is suitable for mobile deployment. Training a model that can effectively operate on a computationally inexpensive device like a mobile phone, and on a range of environments that a user may encounter would both be challenges. Android Studio will need to be used for extensive testing.


1. Introduction and Background


The impact of climate change is getting more and more perceptible with each passing day. Its effects range from storms and wildfires to glacier retreats and increasing sea levels. To stop and remove the damage, we must react quickly. Plastic contamination is a significant factor in the issue. According to research, we found only around 9% of the 500 billion plastic bottles we use year are recycled. In this project, we'll investigate the identification of plastic bottles using machine vision.

1.1 The problem you tried to solve

The main problem we tried to solve was identifying a plastic bottle in a livestream containing multiple objects. Specifically, we want to solve the difficulties of automating this process in larger contexts such as factories or industrial settings that require cameras for waste management or introduce machine learning to surveillance of areas for pollution estimates. With regards to waste management, a challenge exists to successfully process and determine whether pollutants are present or if items are potentially recyclable. Detection of plastic bottles in this context is crucial in determining whether waste may be suitable for further pre-processing in a waste management plant. With regards to surveillance, when attempting to tackle pollution in the wild it can be a challenge to both correctly spot and identify waste in different contexts, as well as perform initial surveillance to estimate pollution in a particular area. This is well-suited to plastic bottle detection that can accurately identify bottles in different contexts and provide simple information to a user such as alerting them to a plastic bottle near them or maintaining a count for an area.

1.2 Motivation

The single-use water bottles made of plastic are the subject of this study. Every minute, over a million single-use water bottles are sold worldwide, but only 30% are recycled. The remainder ends up in landfills or the water without decomposing, persisting as hazardous particles that eventually find their way into people's bodies. Both human health and the environment of the Earth will be severely harmed if this goes on for many years. There is a solid motivation to increase the efficacy of current environmental solutions, such as waste management and human clean-up efforts, by harnessing machine learning models. Furthermore, there is a strong incentive to provide a practical solution to deploy an effective deep learning model on an easy-to-use platform, e.g., an Android application, to increase the likelihood that the average environmentally conscious person would adopt automatic plastic-bottle detection when they are next in a context with pollution.

1.3 Application

An essential application of Plastidetection would be determining the ability to recycle large amounts of waste. Creating an automatic method for classifying plastic bottles is the main problem of this project. This will identify the plastic bottle from the visuals we provide.
This project aims to develop a model that can be paired with a mobile app that can accurately detect multiple plastic bottles that may be present in live video feeds. Concerning a waste management context, the app could be paired with feed from a waste management facility or video recordings of waste disposal to confirm if plastic bottles are present. This would be useful for correct waste sorting or maintaining data about the number of bottles provided (as would be of interest at a recycling centre). 
For environmental pollution, this app would benefit individual users who could provide images or video footage of different areas containing plastic bottles. Key challenges that exist include
•	detecting plastic bottles in a range of contexts based on images that users may provide,
•	dealing with different lighting and conditions, and
•	running a model with moderately high frames per second on a mobile device.
This task will likely require exploring models with high performance on the edge and other low-powered devices to achieve good performance.

1.4 Dataset

The plastic bottle dataset comprises roughly 4,000 YOLO (You Only Look Once) bounding box-annotated photos of plastic bottles that were taken in diverse outdoor areas. The dataset is intended for object detection model training and testing that can recognise plastic water bottles in realistic conditions. Because of the variety in the images, the dataset is both problematic and realistic, making it an essential tool for object detection in research and development. A benefit of this dataset is that images were taken of different bottles from different countries (based on labelling), improving the range of contexts in which images can be taken.
The data has been divided into training, testing, and validation to enable training and testing. The testing and validation set contain 15% and 15% of the total number of images, with the training set making up 70%. This allows the researcher to optimise their methods and provides for a thorough examination of prediction accuracy.


  
2 Overview of the architecture/system

Two training pipelines will be explored to develop an object detection model effectively. The first is the YOLO (You Only Look Once) object detection pipeline, including version 5 and its recent release of version 8. This will be an incredible architecture to explore to increase the detection model’s effectiveness at high frame rates on lower power devices such as mobile phones for processing video and live stream data (Diwan et al., 2022). Both versions, five and eight, have been selected for exploration to compare their accuracy and avoid any compatibility issues that may arise when attempting to deploy the very recent YOLOv8 release. Transfer learning is planned by using YOLO’s provided weights pre-trained on the COCO dataset.
The second training pipeline to be explored is Faster-RCNN which deviates from YOLO by pairing a region proposal network with a CNN feature extractor to form bounding boxes (Ren et al., 2015; He et al., 2015). This differs from YOLO’s ‘grid-based’ approach to object detection for declaring bounding boxes in spatial regions (Abonia Sojasingarayar, 2022). Furthermore, Faster-RCNN allows for more straightforward customization of its CNN backbone parameters, aiding experimentation with different architectures.

2.1 CNN Architecture Design

YOLO Common Components
As Yolov8 is an iteration of Yolov5, the critical components of their architectures are essentially the same. The backbone is responsible for taking the input image and generating feature maps from it, forming the basis of a model. The primary structure involved in both gathering contextual information from the input image and separating the data into patterns makes this a vital stage in any object detector. The structure connecting the head and the backbone that collects as much data as possible before it is sent to the head is called the "neck." By limiting the loss of small-object knowledge to higher levels of abstraction, this structure is crucial in the transfer of that information. For distinct layers from the backbone to be aggregated and recover impact on the detection step, it accomplishes this by extending the size of the feature maps.

Yolov8

The Yolov8 Architecture can be divided into several key sections. The YOLOv8- system backbone takes advantage of a CSPDarknet53 CNN to use as an initial feature extractor. The model has five detection components and a prediction layer for outputting final bounding box decisions from the detection processes. Numerous object detection benchmarks have demonstrated that the YOLOv8 model achieves state-of-the-art performance while retaining high speed and economy.(Jocher, 2023)

Yolov5 – Final Architecture

The Yolov5 backbone also uses the CSP-Darknet53 CNN for initial feature extraction. In the same fashion as Yolov8, it takes advantage of 3 convolution layers to make final predictions as the head of the model. PANet is used as an intermediate step between these stages to serve as the neck of the Yolov5 model. This net allows for bottom-up path augmentation to shorten the path between information and the model layers to simplify the prediction output (Liu et al., 2018). The diagram below provides a visualization of the default architecture with customized CSP bottleneck layers in-between rounds of convolution (Benjumea et al., 2023).(Katsamenis et al., 2022)

Faster-RCNN

Faster R-CNN comprises three main components: a CNN feature extractor similar to YOLO’s approach. Its second component is a region proposal network (RPN) that can divide the final output of the feature extractor into critical regions and predict where the bounding box should be for any objects (Achraf Khazri, 2019). The last element of the Fast-RCNN architecture is a series of fully connected layers that can translate these bounding boxes into accurate predictions like a traditional neural network.(Achraf Khazri, 2019)


2.2 GUI Design
Type of GUI planned: 
We planned to develop an Android application as a user interface.
There are four pages in our application. 
1. Home page: Here, you will find general information about our vision to deploy the app and the Join Us button. This button will lead the user to the sign-in page. Also, an about us button will lead to our team information and contact information.
2. Sign-in page: On this page, the user will find two options for login. He can log in with custom credentials or social logins such as Google, Facebook, and Twitter.
3. Detection page: Upon successful login, the user will prompt to the detection page, where the camera will start, and the user can detect plastic bottles in real time.
4. About Us page: This includes our team and contact information.



3 Results and Evaluation

For initial experimentation 3 main frameworks were used – Yolov5, Yolov8, and Faster R-CNN. Detailed descriptions of their settings are provided below.

3.1 Experimental Settings

Yolov5

Our Yolov5 model was trained using its default Yolov5 version 6 architecture using a CSP Darknet 53 CNN (Convolutional Neural Network) backbone. Transfer learning was used by taking the best state of a previous model that had used fewer epochs.
Parameter	Justification
Epochs = 300	This struck a balance between sufficient model accuracy and efficient training time. Current training took approximately 15 hours. Epochs were raised from 100 to 300 once the model was selected as our final architecture.
Image Size = 640	A size of 640 was selected for the image size as this was the size of images provided by the coco dataset. In testing with a lower 400 value this was able to be scaled sufficiently without being computationally expensive on ram.
Batch Size = 8	Batch sizes were experimented with starting at 64 and halving the value until one sufficiently met system requirements. 9 was the maximum available based on the image size and RAM provided.
Weights = yolov5l.pt	Due to the small size of our dataset, pretrained weights were selected using the provided Yolov5 model to improve the overall accuracy of training. The large model was taken advantage of due to the significant variety in our data to attempt to improve accuracy

Yolov8

Our Yolov8 was also trained with no changes made to the default architecture. This includes the most recent Yolov5 Darknet CSP as its backbone, as well as several changes its convolutional layering (Jocher, 2023).
Parameter	Justification
Epochs = 100	100 epochs were selected to ensure the model achieved sufficient training time to converge to a high precision. This limit was placed so that sufficient early experimentation could be done without exhaustively training each model.
Image Size = 640	Image size was kept as 640 as Yolov8 was also trained on the coco dataset with this image size.
Batch Size = 16	Batch size was set to 16 for training as it was once again the largest that could be handled by the online environment.
Optimizer = ‘SGD’	SGD was selected as an optimizer due to its ability to provide a more generalized final model (Keskar, Nitish Shirish & Socher, 2017). Adam and AdamW are to be explored in the future to compare their performance and training time.
Weights = yolov8l.pt	Pre-trained weights were additionally used for this model as the dataset is not particularly large and would likely suffer in performance if trained from scratch. The large model was once again used due to the significant variety in our data to attempt to improve accuracy

Faster R-CNN

For the final experiment Faster R-CNN was used as an alternative to the Yolo family of frameworks. This was paired with a ResNet50 for a combination of high accuracy and reasonable training time, with higher families taking significant lengths to train sufficient epochs (Deng et al., 2018).
Parameter	Justification
Epochs = 50	Epochs were capped at 50 but are being re-trained with 100. This is due to the significant training time for the model as 50 epochs has currently taken 31 hours to train.
Model = Resnet50	ResNet50 was selected as the model for a balance of both its accuracy and efficient training time. Experiments were also conducted with DarkNet and EfficientNet which were able to slightly improve training time but had lower final precision.
Batch Size = 8	Batch size was raised from 4 to 8 as this was still comfortably managed with the system RAM. Higher values were attempted but were unsuccessful.

3.2 Pre-Processing (If Applicable)

Limited pre-processing was applied to the dataset where necessary. As the dataset already came with a split in the YOLO format for its labelling, this significantly reduced the pre-processing needed. The following was conducted:

-	The train/test/split was adjusted from 60/20/20 to 70/15/15 to improve the amount of training data for the model
-	Several images (<50) had labels missing, and these were manually drawn in and added using the tool LabelMe
-	The dataset was duplicated with labels in Pascal-VOC format to suit Faster-RCNN

3.3 Experimental Results

CNN Framework	Train Precision	Validation Precision	Test Precision
Yolov5	74.291%	71.9%	67.7%
Yolov8	76.7%	76.9%	67.7%
Faster R-CNN	37.0%	30.55%	35.6%

3.4 Limitations

While it may seem that the natural conclusion from the results was to deploy Yolov8 as our model, this had several compatibility issues deploying in tflite format for our Android application. As a substitute, our Yolov5 model was deployed with no problem and a solid ability to translate its detections in real time using a mobile phone camera. This was a significant limitation, and however fortunately, these models had similar identical test accuracy, which meant only a little was sacrificed to continue with this deployment. A limitation of our decision to use an Android application in the first place means that our target audience is mainly those who may be operating the application outdoors or conducting surveillance about pollution in a particular area. Another fundamental limitation is the size of bottles present in the image – as the dataset broadly includes more explicit and close-up photos of bottles, the detection rate on those that may be obscured outdoors or in extremely bright or dark lighting is significantly lower.
We kept the default login credentials in the Android application as “admin”. We didn’t integrate the signup process because of database limitations. Also, our application uses only real-time streaming.



4 Discussion

Accuracy amongst all models was relatively low overall as the dataset contains many plastic bottles in widely different contexts. This overall made it challenging to provide a firm number of sample sizes in each environment and unpredictable to test. The train/test/split provided was also done randomly, so there is a risk of over-representing images in the test set. 
Yolov8 provided the highest accuracy, marginally outperforming yolov5 across training, validation, and testing. This is likely due to its improved framework architecture and feature mapping, which increases its ability to make more complex decisions and generalizations – this has been quite useful given the wide variety in our dataset (Jocher, 2023). Both models saw notably lower accuracy on the test dataset. They may benefit from an improved model, such as adding dropout layers to the architecture or fine-tuning the regularization hyper-parameters to prevent a risk of overfitting. Both models share similar results when making detections on images. Figures attached in their results show example misclassifications and correct detections. The most common example that degrades accuracy is drawing a poor-quality bounding box or one with a significant background. This is primarily due to the variety of backgrounds and contexts in the dataset, making it challenging to determine bottles in particular circumstances.
Faster-RCNN demonstrated significantly low accuracy, and insufficient epochs were likely provided for the model to converge correctly. The model accuracy showed no signs of an early plateau and would almost certainly benefit from extended training. The model is being retrained for 100 epochs and using the alternate DarkNet backbone to compare its performance to ResNet50. With this, the precision of the model can be improved as it is doubtful that this is the limit of Faster-RCNN's performance. Example inferences have been displayed above for the model that shows predictions. The most common misclassification the model tends to undergo is drawing excess bounding boxes that include a lot of background or multiple bounding boxes for only one object.
We first decided to use yolov8 in tflite format for Android app development. Still, as it was newly launched approximately 3-4 months back, we needed more resources to deploy it successfully in Android. Hence, we trained yolov5 with higher epochs and used tflite format of the trained model in Android. As the test accuracy of v5 and v8 are the same, our application shows promising results with detection.
