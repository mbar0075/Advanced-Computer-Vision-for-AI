# Advanced Computer Vision for AI

<p align="justify">

The research delved into multiple facets of advanced computer vision, exploring **activation functions**, **CNN refinement** for CIFAR10, **transfer learning**, **fine-tuning**, and **object detection**. In assessing activation functions, ReLU emerged as dominant, while optimal learning rates and network complexity significantly influenced convergence. The adapted CNN architecture for CIFAR10 remarkably achieved 82% across accuracy, precision, recall, and F1-Score, efficiently leveraging 531,818 parameters. Notable strategies included expanded layers, diverse activation functions, and optimization techniques. Transfer learning and fine-tuning highlighted the importance of learning rates and epochs, with AdaMax and Adam optimizers excelling. Model architecture evaluation favored MobileNetV2 for image classification. Object detection comparisons identified YOLOv8 as the most proficient, excelling in precision and recall.

## Task 1: Understanding Activation Functions

This research delves into the exploration of activation functions within the neural network framework. The study revealed ReLU as the predominant performer across various datasets, showcasing its efficacy in enabling convergence. Notably, learning rates between 0.1 and 1 were identified as optimal for achieving convergence on the Circle dataset, while extremes led to output oscillations. Additionally, variations in network layers and neurons influenced decision boundary complexity and convergence speed, highlighting the crucial balance needed in model complexity. Evaluation of optimization algorithms highlighted Adadelta's superiority, exhibiting higher training and testing accuracy, along with reduced loss in comparison to other methods.

<p align='center'>
  <img src="Individual Assignment/Assignment 1/neural_network_1_layer.png" alt="TensorFlow Playground">
</p>

<p align='center'>
  <img src="Individual Assignment/Assignment 1/evaluation_graph.png" width="90%" alt="Evaluation Graph">
</p>

## Task 2: Understanding Convolutional Neural Networks

The study focused on refining TensorFlow's CNN tutorial for improved performance on the `CIFAR10` dataset. CIFAR10 comprises 60,000 color images categorized into `10` classes, each containing 6,000 images. This dataset is split into 50,000 training images and 10,000 testing images. Each image is a 32x32 pixel array, making it a challenging yet standard benchmark in the field of computer vision due to its diverse classes and low resolution.

<p align='center'>
  <img src="Individual Assignment/Assignment 2/data.png" alt="CIFAR10 Dataset" width="70%">
</p>

The adapted CNN architecture achieved remarkable enhancements, attaining an outstanding **82%** `Accuracy`, `Precision`, `Recall`, and `F1-Score` on the CIFAR10 test set. These improvements were made while judiciously utilizing 531,818 model parameters, showcasing the efficiency and optimization achieved through the adapted architecture.

### Proposed Enhanced CNN Architecture for the CIFAR10 Dataset
<p align="center">

| No.   | Layer Type           | Details                |
|-------|----------------------|------------------------|
| 1     | `Conv2D`             | (3, 3), 32 filters, `ELU` |
| 2     | `BatchNormalization` | -                      |
| 3     | `Conv2D`             | (3, 3), 32 filters, `ELU` |
| 4     | `BatchNormalization` | -                      |
| 5     | `MaxPooling2D`       | (2, 2)                 |
| 6     | `Dropout`            | 25%                    |
| 7     | `Conv2D`             | (3, 3), 64 filters, `ELU` |
| 8     | `BatchNormalization` | -                      |
| 9     | `Conv2D`             | (3, 3), 64 filters, `ELU` |
| 10    | `BatchNormalization` | -                      |
| 11    | `MaxPooling2D`       | (2, 2)                 |
| 12    | `Dropout`            | 25%                    |
| 13    | `Conv2D`             | (3, 3), 128 filters, `ELU` |
| 14    | `BatchNormalization` | -                      |
| 15    | `Conv2D`             | (3, 3), 128 filters, `ELU` |
| 16    | `BatchNormalization` | -                      |
| 17    | `Dropout`            | 25%                    |
| 18    | `Flatten`            | -                      |
| 19    | `Dense`              | 512 neurons, `ReLU`    |
| 20    | `BatchNormalization` | -                      |
| 21    | `Dropout`            | 50%                    |
| 22    | `Dense`              | 256 neurons, `ReLU`    |
| 23    | `BatchNormalization` | -                      |
| 24    | `Dropout`            | 50%                    |
| 25    | `Dense`              | 128 neurons, `ReLU`    |
| 26    | `BatchNormalization` | -                      |
| 27    | `Dropout`            | 50%                    |
| 28    | `Dense`              | 64 neurons, `ReLU`     |
| 29    | `BatchNormalization` | -                      |
| 30    | `Dropout`            | 50%                    |
| 31    | `Dense`              | 10 neurons             |
| 32    | `Softmax`            | -                      |

</p>

The architectural enhancements focused on expanding convolutional and dense layers, incorporating optimal activation functions like softmax, ELU, and ReLU, and employing optimization techniques such as the Adam optimizer with gradient clipping. Additional strategies included integrating dropout and batch normalization layers, utilizing a larger batch size and epochs, and incorporating an early stopping callback.


<p align='center'>
<table align="center">
  <tr>
    <td align="center"  width="50%">
      <img src="Individual Assignment/Assignment 2/best_cnn_model.png" alt="Validation Graphs"/>
    </td>
    <td align="center" width="50%">
      <img src="Individual Assignment/Assignment 2/confusion_matrix.png" alt="Confusion Matrix"/>
    </td>
  </tr>
</table>
</p>

Observations from the enhanced model revealed notable outcomes: training and validation curves displayed gradual convergence, indicating minimized overfitting. However, challenges in accurately classifying specific classes, notably `cat` and `dog`, were identified from the confusion matrix. Despite this, a distinct diagonal line in the matrix highlighted the model's generally accurate classification across most classes.

### **CIFAR DecaLuminarNet:** A Further Enhanced CNN Architecture for CIFAR10

The proposed architecture was further refined to achieve a more efficient model. The adapted architecture, named `CIFAR DecaLuminarNet`, achieved an impressive **87%** `Accuracy`, `Precision`, `Recall`, and `F1-Score` on the CIFAR10 test set, utilizing **9,136,778** parameters. This model was trained for **150** epochs with a batch size of **512**. The `DecaLuminarNet` conveys deep and illuminating insights into the model's architecture, with `Deca` representing the 10 classes in CIFAR10 and `LuminarNet` signifying the model's illuminating performance over the first proposed architecture.
<p align="center">

| No.   | Layer Type           | Details                |
|-------|----------------------|------------------------|
| 1     | `Conv2D`             | (3, 3), 64 filters, `ELU`, `padding='same'` |
| 2     | `BatchNormalization` | -                      |
| 3     | `Conv2D`             | (3, 3), 64 filters, `ELU`, `padding='same'` |
| 4     | `BatchNormalization` | -                      |
| 5     | `MaxPooling2D`       | (2, 2)                 |
| 6     | `Dropout`            | 25%                    |
| 7     | `Conv2D`             | (3, 3), 128 filters, `ELU`, `padding='same'` |
| 8     | `BatchNormalization` | -                      |
| 9     | `Conv2D`             | (3, 3), 128 filters, `ELU`, `padding='same'` |
| 10    | `BatchNormalization` | -                      |
| 11    | `MaxPooling2D`       | (2, 2)                 |
| 12    | `Dropout`            | 25%                    |
| 13    | `Conv2D`             | (3, 3), 256 filters, `ELU`, `padding='same'` |
| 14    | `BatchNormalization` | -                      |
| 15    | `Conv2D`             | (3, 3), 256 filters, `ELU`, `padding='same'` |
| 16    | `BatchNormalization` | -                      |
| 17    | `Conv2D`             | (3, 3), 256 filters, `ELU`, `padding='same'` |
| 18    | `BatchNormalization` | -                      |
| 19    | `MaxPooling2D`       | (2, 2)                 |
| 20    | `Dropout`            | 25%                    |
| 21    | `Flatten`            | -                      |
| 22    | `Dense`              | 512 neurons, `ReLU`    |
| 23    | `BatchNormalization` | -                      |
| 24    | `Dropout`            | 50%                    |
| 25    | `Dense`              | 512 neurons, `ReLU`    |
| 26    | `BatchNormalization` | -                      |
| 27    | `Dropout`            | 50%                    |
| 28    | `Dense`              | 256 neurons, `ReLU`    |
| 29    | `BatchNormalization` | -                      |
| 30    | `Dropout`            | 50%                    |
| 31    | `Dense`              | 128 neurons, `ReLU`    |
| 32    | `BatchNormalization` | -                      |
| 33    | `Dropout`            | 50%                    |
| 34    | `Dense`              | 64 neurons, `ReLU`     |
| 35    | `BatchNormalization` | -                      |
| 36    | `Dropout`            | 50%                    |
| 37    | `Dense`              | 10 neurons, `Softmax`  |

</p>

The refined architecture focused on expanding convolutional layers, incorporating renowned optimal activation functions like softmax and ELU, and employing optimization techniques such as the Adam optimizer with gradient clipping. Additional strategies included integrating dropout and batch normalization layers, utilizing a larger batch size and epochs, and incorporating an early stopping and reduce learning rate on plateau callbacks. 

<p align='center'>
<table align="center">
  <tr>
    <td align="center" width="52%">
      <img src="Individual Assignment/Assignment 2/cifar_decaluminar_net_model.png" alt="Validation Graphs"/>
    </td>
    <td align="center" width="50%">
      <img src="Individual Assignment/Assignment 2/cifar_decaluminar_net_confusion_matrix.png" alt="Confusion Matrix"/>
    </td>
  </tr>
</table>
</p>

Observations from the enhanced model revealed notable outcomes: training and validation curves displayed gradual convergence, indicating minimized overfitting. However, challenges in accurately classifying specific classes, notably `cat` and `dog` persisted. Despite this, a distinct diagonal line in the matrix highlighted the model's generally accurate classification across most classes.

## Task 3: Transfer Learning & Fine-Tuning

This research explored advanced computer vision techniques, including transfer learning, fine-tuning, learning rates, optimizers, and model architectures, aiming to enhance image classification. The study aimed to identify crucial elements in the classification process and assess the combined impact of these techniques on overall performance.

<p align='center'>
  <img src="Individual Assignment/Assignment 3/data.png" alt="Cats and Dog Dataset" width="70%">
</p>

### Learning Rate & Epochs
Smaller learning rates, particularly 0.001, demonstrated superior initial performance with the `Cats and Dogs` dataset, resulting in higher validation accuracy and lower loss. Extremes (0.1 and 0.0001) led to poorer outcomes. Increasing the number of epochs notably improved model convergence and validation accuracy.

<p align='center'>
<table align="center">
  <tr>
    <td align="center">
      <img src="Individual Assignment/Assignment 3/plots/Validation Accuracy and Loss for different learning rates.png" alt="Validation Graphs LR"  width="100%" />
    </td>
    <td align="center">
      <img src="Individual Assignment/Assignment 3/plots/Validation Accuracy and Loss for different epochs.png" alt="Validation Graphs Epochs"  width="100%" />
    </td>
  </tr>
</table>
</p>

### Fine-Tuning & Optimizers
AdaMax and Adam optimizers showcased superior performance due to their adaptive learning rate strategies, resulting in higher accuracy and reduced loss. Conversely, AdaDelta's limitations in capturing intricate dataset patterns led to inferior performance. Additionally, repeated performance jumps during fine-tuning suggested instability in model performance.

<p align='center'>
<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="Individual Assignment/Assignment 3/plots/Validation Accuracy and Loss for different optimizers.png" alt="Validation Graphs Optimizers"/>
    </td>
    <td align="center" width="36%">
      <img src="Individual Assignment/Assignment 3/plots/Adamax.png" alt="Validation Graphs Epochs"/>
    </td>
  </tr>
</table>
</p>

### Model Architectures
Various model architectures were evaluated for image classification. MobileNetV2 displayed strong initial validation accuracy (Val Acc) of **0.9678** and excelled in fine-tuning with a Val Acc of **0.9913**, achieving the highest Test Acc of **0.9948**. ResNet50 and EfficientNetB0 emerged as noteworthy competitors, showcasing high accuracy and low loss, while InceptionV3 displayed the lowest validation loss.

<p align='center'>
  <img src="Individual Assignment/Assignment 3/plots/Model Architecture Table.png" width="80%" alt="Model Architecture Table">
</p>

1. **Static Params:** **Loss** = Cross Entropy, **LR** = 0.001, **Initial Epochs** = 10, **Fine-Tune Epochs** = 10, **Base Model** = MobileNetV2
2. **Static Params:** **Loss** = Cross Entropy, **LR** = 0.001, **Initial Epochs** = 10, **Fine-Tune Epochs** = 10
3. **Static Params:** **Loss** = Cross Entropy, **LR** = 0.001, **Initial Epochs** = 10, **Fine-Tune Epochs** = 10

## Task 4: Pizza Object Detection
### Introduction
In this project, object classification models like **ResNet50**, **VGG16**, and **MobileNet** were employed to classify images into two categories: those containing pizza and those that didn't. Subsequently, an annotated dataset encompassing diverse pizza types and ingredients was created using **RoboFlow**. This dataset was then used to train object-detection models—**YOLOv8**, **YOLOv5**, **RetinaNet**—aimed at identifying pizzas and their ingredients. Loss and accuracy graphs were generated for evaluation. Although a basic implementation of **DETR** was explored, it was not included in the final assessment. The culmination of these techniques led to the construction of a comprehensive pizza analysis system.
<br><br>

<p align='center'>
  <img src="Group Project Assignment/Diagrams/pipeline.drawio.png" alt="Pipeline">
</p>

### Setup

The code for this research was executed on Google Colab, with provided environment files (`environment.yml` and `requirements.txt`) for local execution of object detectors. The main directory structure includes the following folders:

- **Papers:** Contains references to any papers cited in the documentation.
- **Part_1_Object_Classification:** Includes files pertinent to the Object Classification task.
- **Part_2_Building a Dataset:** Encompasses files related to the creation of an annotated dataset.
- **Part_3_Object_Detection:** Holds files related to the Object Detection task.
- **Part_3_Results:** Consists of diagrams or images created for evaluation of both object classification and object detection tasks.
- **pizza_data:** Contains the Kaggle Pizza Data Dataset [1].
- **pizza_classification:** Holds the Kaggle Pizza Classification Dataset [2].

<!-- <p align='center'>
  <img src="Assets/fileexplorerMain.png" alt="File Explorer">
</p> -->

### Object Classification

Object classification is a core task in computer vision, involving the detection and categorization of objects based on predefined classes. Models trained on labeled datasets can assign labels to images with varying confidence levels, often requiring a threshold to filter low-confidence predictions.

<p align='center'>
<table align="center">
  <tr>
    <td align="center">
      <img src="Group Project Assignment/Assets/Object Classification/resnet50.png" alt="ResNet50"  width="100%" height="auto" />
    </td>
    <td align="center">
      <img src="Group Project Assignment/Assets/Object Classification/vgg16.png" alt="VGG16" width="100%" height="auto" />
    </td>
    <td align="center">
      <img src="Group Project Assignment/Assets/Object Classification/mobilenet.png" alt="MobileNet" width="100%" height="auto" />
    </td>
  </tr>
</table>
</p>

#### ImageNet
ImageNet (2009) is a widely used dataset for training computer vision models, though not directly utilized in this project. Pre-trained models with ImageNet weights were employed.

#### VGG16
Introduced in 2014, VGG16 excels in image classification despite its relative simplicity compared to newer architectures, focusing on capturing intricate patterns within images.

#### ResNet50
ResNet50 (2015) addresses deep network training issues using residual learning blocks and skip connections, enhancing accuracy in image recognition by overcoming the vanishing gradient problem.

#### MobileNet
Designed in 2017 for mobile and edge devices with limited computational resources, MobileNet employs depth-wise separable convolutions, striking a balance between accuracy and efficiency for real-time applications.

Three classification models—ResNet50, VGG16, and MobileNet—were chosen for implementation in this study. Apart from their popularity, these models were selected due to their manageable computational requirements and diverse architectures. A Kaggle-retrieved dataset [2] was used for comparison purposes. Images were resized to 224x224 to accommodate model input size requirements, ensuring fair evaluation among the architectures.

<p align='center'>
  <img src="Group Project Assignment/Assets/Object Classification/comparisons_classification.png" alt="Comparisons">
</p>

Utilizing pre-trained weights from ImageNet, the classification process commenced by feeding images to each model. The predict function returned lists of tuples containing label identifiers, names, and associated confidence scores for each image. Top five labels per image, ranked by confidence, were displayed and saved into JSON files corresponding to each model used.

Despite their capability to classify various labels, these models were limited to classifying a pizza without distinguishing its toppings, as none were part of the known label list.

<p align='center'>
<table align="center">
  <tr>
    <td align="center">
      <img src="Group Project Assignment/Assets/Object Classification/resnet50CM.png" alt="ResNet50"  width="100%" height="auto" />
    </td>
    <td align="center">
      <img src="Group Project Assignment/Assets/Object Classification/vgg16CM.png" alt="VGG16" width="100%" height="auto" />
    </td>
    <td align="center">
      <img src="Group Project Assignment/Assets/Object Classification/mobilenetCM.png" alt="MobileNet" width="100%" height="auto" />
    </td>
  </tr>
</table>
</p>

### Dataset Creation

Building a robust annotated dataset is a time-consuming process vital for object detection. It begins with dataset selection and label definition before the annotation process. Bounding boxes were used for this project, while other tasks might employ polygon annotation. Precise and consistent annotations are essential, facilitated by tools like Roboflow.

#### Roboflow
Established in 2019, Roboflow is a user-friendly platform offering annotation, augmentation, and organization tools, aiding efficient dataset management.

#### Annotated Pizza Dataset
For object detection, a Kaggle-sourced pizza dataset of around `9000` images was utilized [1]. Approximately `1500` images were annotated, focusing on `16` common pizza ingredients for labeling.

Chosen ingredient labels:
1. Arugula
2. Bacon
3. Basil
4. Broccoli
5. Cheese
6. Chicken
7. Corn
8. Ham
9. Mushroom
10. Olives
11. Onion
12. Pepperoni
13. Peppers
14. Pineapple
15. Pizza
16. Tomatoes

To streamline annotation and system performance, certain ingredients were omitted. Roboflow's tools were used for label management, bounding box creation, and image sorting.

#### Data Preparation
Following annotation, the dataset was split into training, validation, and testing sets in a `60%-20%-20%` ratio. Augmentations, like rotation and blur, were exclusively applied to the training set, resulting in `2544`, `284`, and `283` images, respectively. The dataset was exported in various formats via Roboflow to suit object detection model requirements.

<p align='center'>
  <img src="Group Project Assignment/Assets\dataset_split.png" alt="Dataset Split">
</p>

This meticulous dataset preparation laid the foundation for subsequent training and evaluation phases of the object detection model.

#### Additional Roboflow Model (Roboflow 3.0)

Roboflow introduced its own model (**Roboflow 3.0**) utilizing the labeled dataset mentioned earlier for training. The dataset is available at https://app.roboflow.com/advanced-computer-vision-assignment/pizza-object-detector/deploy/7, while the resultant model can be accessed via the QR code or through the deployment dashboard displayed below.

<p align='center'>
<a href="https://app.roboflow.com/advanced-computer-vision-assignment/pizza-object-detector/deploy/7">
  <img src="Group Project Assignment/Assets/Roboflow_deployment.png" alt="Roboflow Deployment">
</a>
</p>

The QR code can be scanned to access the deployment dashboard below:
<p align='center'>
<a href="https://app.roboflow.com/advanced-computer-vision-assignment/pizza-object-detector/deploy/7">
  <img src="Group Project Assignment/Assets/QR Code.png" alt="Roboflow Deployment" width="20%" height="auto">
</a>
</p>

To cite the dataset, please use the following BibTeX format:

```bib
@misc{ pizza-object-detector_dataset,
    title = { Pizza Object Detector Dataset },
    type = { Open Source Dataset },
    author = {Matthias Bartolo and Jerome Agius and Isaac Muscat},
    howpublished = { \url{ https://universe.roboflow.com/advanced-computer-vision-assignment/pizza-object-detector } },
    url = { https://universe.roboflow.com/advanced-computer-vision-assignment/pizza-object-detector },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2023 },
    month = { nov },
}
```

### Object Detection

Object Detection in computer vision involves identifying objects and their locations within images or videos, incorporating bounding boxes and class labels. Unlike object classification, it pinpoints an object's position through bounding boxes and assigns appropriate class labels.

#### RetinaNet
RetinaNet (2017) addresses class imbalance and localization with its Focal Loss, emphasizing difficult examples during training. Its feature pyramid network handles multi-scale features, making it proficient in complex, multi-scale object detection scenarios.

<p align='center'>
<table align="center">
  <tr>
    <td align="center" width="63%">
      <img src="Group Project Assignment/Part_3_Results/RetinaNet/inference_images/000001_jpg.rf.28a0a9ec43243edc5b2179d657363422.jpg.jpg" alt="Image 1" />
    </td>
    <td align="center" width="50%">
      <img src="Group Project Assignment/Part_3_Results/RetinaNet/inference_images/00151_jpg.rf.575d4c0db587e5567e95e15f32fb479d.jpg.jpg" alt="Image 2"/>
    </td>
  </tr>
</table>
</p>


#### YOLOv5
YOLOv5 (2020) by Ultralytics improves YOLO's legacy with a more streamlined architecture, offering customization and multi-platform deployment. Variants like YOLOv5x and YOLOv5s cater to different computational requirements, focusing on simplicity, efficiency, and performance.

<p align='center'>
<table align="center">
  <tr>
    <td align="center" width="38%">
      <img src="Group Project Assignment/Part_3_Results/YOLOv5/results/detect/exp/00061_jpg.rf.d91f39fda40d6ef026acb7a15751f913.jpg" alt="Image 1"/>
    </td>
    <td align="center" width="50%">
      <img src="Group Project Assignment/Part_3_Results/YOLOv5/results/detect/exp/01167_jpg.rf.913f261297783a8efd10ca65fd78a429.jpg" alt="Image 2"/>
    </td>
  </tr>
</table>
</p>

#### YOLOv8
YOLOv8 (2023) continues advancements, utilizing CSPDarknet53 for efficient image processing while maintaining high accuracy. Variants like YOLOv8-CSP and YOLOv8-Darknet provide options for different computational resources and use cases.

<p align='center'>
  <img src="Group Project Assignment/Assets/YOLOv8Examples.jpg" alt="YOLOv8">
</p>

#### DETR
DETR (2020) from Facebook AI Research replaces traditional anchor-based methods with transformers for end-to-end object detection. It handles multiple objects effectively in complex environments, especially with small-sized objects.

<p align='center'>
<table align="center">
  <tr>
    <td align="center">
      <img src="Group Project Assignment/Part_3_Results/DETR/groundTurth.png" alt="Ground Truth"  width="100%"/>
    </td>
    <td align="center">
      <img src="Group Project Assignment/Part_3_Results/DETR/Prediction.png" alt="DETR Prediction"  width="100%"/>
    </td>
  </tr>
</table>
</p>

The project implemented YOLOv5, YOLOv8, RetinaNet, and DETR for object detection, leveraging their specific strengths. Each model underwent dataset preparation, training, and evaluation. Post-training, examples of object detection in test images were showcased to assess model performance and capabilities.

### Evaluation

The evaluation compared YOLOv5, YOLOv8, RetinaNet, and DETR models. YOLOv8 showcased superior overall performance compared to YOLOv5, demonstrating better precision and recall in most scenarios despite the use of early stopping. While YOLOv5 displayed more consistent gradient reduction and higher precision at greater recall levels in precision-recall curves, YOLOv8 emerged as the best-performing model considering multiple evaluation metrics and curves. RetinaNet, an older architecture, showed lower precision but higher recall, producing acceptable results compared to modern YOLO architectures. Though DETR was implemented without specific graphs, a table with Average Precision and Recall values was provided, enabling comparison against other models. In summary, while YOLOv5 excelled in certain aspects, the comprehensive evaluation favored YOLOv8 as the top-performing model in this evaluation.

<p align='center'>
<table align="center">
  <tr>
    <td align="center">
      <img src="Group Project Assignment/Part_3_Results/RetinaNet/RetinaNet MAP.png" alt="RetinaNet MAP"  width="100%" />
      <strong>RetinaNet MAP Graph</strong>
    </td>
    <td align="center">
      <img src="Group Project Assignment/Part_3_Results/RetinaNet/RetinaNetPR.png" alt="RetinaNet PR"  width="100%"/>
      <strong>RetinaNet PR Curve</strong>
    </td>
  </tr>
</table>
</p>

<p align='center'>
<table align="center">
  <tr>
    <td align="center">
      <img src="Group Project Assignment/Part_3_Results/YOLOv5/results/val/yolov5s_results2 (testing)/confusion_matrix.png" alt="YOLOv5 CM"  width="100%"/>
      <strong>YOLOv5 Confusion Matrix</strong>
    </td>
    <td align="center">
      <img src="Group Project Assignment/Part_3_Results/YOLOv8/detect/validation/confusion_matrix_normalized.png" alt="YOLOv8 CM"  width="100%"/>
        <strong>YOLOv8 Confusion Matrix</strong>
    </td>
  </tr>
</table>
</p>

### Conclusion
The evaluation highlighted VGG16 as the top-performing image classifier, achieving the highest precision and F1-Score among the implemented models. For object detection tasks on a custom annotated pizza dataset, YOLOv8 emerged as the best-performing model, exhibiting superior curves compared to others. This study underscores the performance of various models in image classification and object detection tasks using a custom annotated dataset. While certain models excelled in this specific dataset, results could have varied with a different dataset selection. Additionally, the choice of pre-defined labels might have impacted the performance of object detection models.

### References

[1] M. Bryant, Pizza images with topping labels,
https://www.kaggle.com/datasets/michaelbryantds/pizza-images-with-topping-
labels/, Jun. 2019.

[2] Project_SHS, Pizza images with topping labels,
https://www.kaggle.com/datasets/projectshs/pizza-classification-data/, Dec.
2022.

</p>

<!-- The dataset which was used can be found through the following link: https://www.kaggle.com/datasets/michaelbryantds/pizza-images-with-topping-labels/

Classificaiton dataset: https://www.kaggle.com/datasets/projectshs/pizza-classification-data/

GitHub code from which we retrieved the ImageNet pizza images
https://github.com/mf1024/ImageNet-Datasets-Downloader/tree/master -->
