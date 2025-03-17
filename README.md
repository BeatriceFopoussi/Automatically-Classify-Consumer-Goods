# Automatically Classify Consumer Goods

Classifiez automatiquement des biens de consommation
## Project Data Science Problem Statement

On this English-speaking marketplace platform, sellers offer items to buyers by posting a photo and a description. Currently, the assignment of a category to an item is done manually by the sellers, leading to limited reliability. Additionally, the volume of items is still very low.

To enhance the user experience for sellers (by facilitating the listing of new items) and buyers (by simplifying product searches), and in anticipation of scaling up, it becomes essential to automate this category assignment task.

The mission is to conduct a feasibility study to develop an automatic item classification engine, utilizing both the image and the description of the products in the dataset.


The assignment of a category to an item is done manually. 
- Task Automation 
- Missions: 
  - Feasibility study of an article classification engine based on an image and a description 
  - Perform supervised classification based on images
* Dataset [https://www.kaggle.com/datasets/atharvjairath/flipkart-ecommerce-dataset]
This is an E-commerce Flipkart Dataset with exactly 20,000 samples. It has 15 columns with a lot of information. You can use is to predict which category it might fall under, considering “Description” for a product, analyze it [https://www.kaggle.com/datasets/atharvjairath/flipkart-ecommerce-dataset]


DIn our dataset, the target columns are:

- The **Description** column, which contains the textual data of product descriptions.
- The **image** column, which contains the images of the products.

To analyze the automatic assignment of classes, the **product_category_tree** column groups the different product categories.

Our goal is to automate the process of assigning a description or an image to a category in the **product_category_tree**.

To achieve this, we use various tools and models:

- For image processing, we leverage architectures such as **MobileNet**, **VGG16**, and **ResNet50**.
- For textual data, we apply techniques like **TF-IDF**, **Word2Vec**, and **BERT** via Hugging Face.

For clustering, we implement algorithms such as **K-means** and **DBSCAN**.

Finally, for cluster visualization and dimensionality reduction, we use methods like **UMAP**, **t-SNE**, and **PCA** (Principal Component Analysis).
