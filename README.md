# CNN-for-Pathology-Report-Classification

One Paragraph of project description goes here

## Introduction

This project presents how to use Covolutional Neural Networks(CNN) to solve Pathology report classification problem. CNNs are always considered as good at processing image data. But this project shows that CNNs can also achieve good performance in Nature Language Processing (NLP). 

In this project, we use pretrained [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) model to convert pathology reports to vector matrix.

### Prerequisities

Required softwares:
```
[neon 1.5.4+485033c](http://neon.nervanasys.com/docs/latest/installation.html)
CUDA 7.5
```
### Result

* Macro=0.175676848187
* Micro=0.479207920792
```
               precision    recall  f1-score   support

          0       0.00      0.00      0.00        30
          1       0.41      0.34      0.37       140
          2       0.00      0.00      0.00        20
          3       0.15      0.07      0.10        80
          4       0.52      0.67      0.59       200
          5       0.00      0.00      0.00        20
          6       0.00      0.00      0.00        40
          7       0.00      0.00      0.00        10
          8       0.32      0.10      0.15        70
          9       0.00      0.00      0.00        30
         10       0.38      0.17      0.24        70
         11       0.52      0.93      0.66       300

avg / total       0.37      0.48      0.40      1010
```

![alt tag](https://github.com/tianxiangchen2015/CNN-for-Pathology-Report-Classification/blob/master/cm_pathology.png)

## Authors

* **Tianxiang Chen (ORNL Research Assistant)** - [Linkedin HomePage](https://www.linkedin.com/in/tianxiang-chen-946543114?trk=nav_responsive_tab_profile)


## Acknowledgments

* **Yoon Kim** - [Github](https://github.com/yoonkim/CNN_sentence)
    
