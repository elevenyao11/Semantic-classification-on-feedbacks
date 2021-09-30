# Semantic-classification-on-feedbacks

## Description

The goal of this project is to help the Canada Digital Analytics Team (CDAT) classify or predict text feedback from users that visit the federal government website (canada.ca) into different tag groups and generate new tags using machine learning models in order to better understand the users’ needs.

We built two multi-class classification models -- Linear SVC and BERT (Devlin et al., 2019) --to predict tags that capture the users’ purpose for visiting the website. In order to provide more insight on users’ requests, we have developed two unsupervised models as well. One is a topic model that uses a RoBERTa architecture (Bhanja, 2020) to generate new tags from users’ feedback, another is a hierarchical clustering model which could cluster the feedback into groups in different levels. To improve the efficiency and performance of our product, we developed linguistic features on both supervised and unsupervised learning models which include named-entity recognition, semantic role labeling, sentiment analysis, and dependency parsing.

By the end of the project, our supervised learning models have achieved 93% accuracy on domain classification and 86% accuracy on vaccine tag classification, which is a 20% improvement compared to the CDAT baseline. Furthermore, our unsupervised learning models have successfully generated new topic groups which could help CDAT better identify the users’ needs. 


The details of the project can be found here: [project plan](/project_plan/project_plan.ipynb)




## Folders

```
.
├── EDA
│   ├── EDA_on_sample_data_of_Canada_CA.md
│   ├── EDA_on_sample_data_of_Canada_CA.docx
│   └── media
├── models
│   ├── preprocessing.ipynb
│   ├── supervised
│   │   ├── CNN.ipynb
│   │   ├── LinearSVC.ipynb
│   │   ├── benefits_bert.ipynb
│   │   ├── taxes_bert.ipynb
│   │   ├── travel_bert.ipynb
│   │   └── vaccine_bert.ipynb
│   ├── unsupervised
│   │   └── Roberta_topic.ipynb
│   └── linguistic_features
│       ├── NER.ipynb
│       ├── parsing.ipynb
│       ├── sentiment.ipynb
│       └── SRL
├── meeting_minutes
├── meetubg_slides
└── project_plan
    └── project_plan.ipynb

```


## Contributors:
- Yundong Yao
- Jan Urquico
- Linxuan Yang
- Alex Chen


