# Automated MLOps Pipeline and Generative AI Applications

This repository contains three distinct but interrelated projects that demonstrate expertise in building scalable AI/ML solutions.

1. **End-to-End MLOps Pipeline for Predicting Student Math Scores**  
2. **Generative AI Application for Blog Creation**  
3. **Hugging Face Summarization Model Fine-Tuning for Messenger Conversations**

All three projects showcase advanced workflows, cloud deployment, orchestration techniques, and a focus on scalability.

---

## **Project 1: End-to-End MLOps Pipeline for Predicting Student Math Scores**

### Overview  
An automated pipeline to predict student math scores using various regression models, including Random Forest and XGBoost, with Linear Regression achieving the best performance (R² = 0.88).

### Key Features  
- **EDA (Exploratory Data Analysis):** Identified significant factors like lunch type and parental education affecting math scores.  
- **Model Training:** Tested multiple models to determine the best fit for accurate predictions.  
- **MLOps:** Built a robust pipeline including preprocessing, model training, and deployment.  
- **CI/CD Integration:** Automated deployment using pipelines.  
- **AWS Deployment:** Deployed on AWS infrastructure for scalability and accessibility.

### Tools & Technologies  
- **MLFlow Tracking:** For experiment tracking and logging.  
- **Apache Airflow DAGs:** For workflow orchestration.  
- **DagsHub Integration:** For collaborative version control and tracking.  
- **AWS Services:** For deployment.

---

## **Project 2: Generative AI Application for Blog Creation**

### Overview  
Developed an AI-driven application to generate blogs using AWS Lambda and Bedrock, leveraging fine-tuned Hugging Face models.

### Key Features  
- **LLMOps Workflows:** Enabled efficient model management and deployment using NVIDIA NeMo.  
- **Scalability:** Integrated AWS infrastructure, including S3 for data storage and Lambda for serverless execution.  
- **Model Fine-tuning:** Optimized Hugging Face models for specific generative tasks.

### Tools & Technologies  
- **AWS Lambda and Bedrock:** For scalable serverless deployment.  
- **S3 Bucket:** For efficient data storage.

---

## **Project 3: Hugging Face Summarization Model Fine-Tuning for Messenger Conversations**

### Overview  
Fine-tuned a Hugging Face summarization model for messenger-like conversations using the **SAMSum dataset**, which contains 16k conversations annotated with concise summaries. This fine-tuned model is specifically tailored to generate summaries of casual, semi-formal, and formal conversations typically found in messaging applications. The model is exposed through a **FastAPI** application, enabling real-time summarization.

### Key Features  
- **Dataset:** The SAMSum dataset contains linguistically annotated conversations, with a range of styles from informal to formal, and including slang, emoticons, and typos.
- **Summarization:** The model generates summaries in a concise, third-person format, capturing the essence of the conversation.
- **FastAPI Integration:** Exposed the fine-tuned model through a FastAPI application for real-time summarization of messenger conversations.
- **Real-Time Summarization:** API endpoints allow users to submit conversation text and receive generated summaries in real-time.

### Tools & Technologies  
- **Hugging Face Transformers:** For pre-trained models and fine-tuning capabilities.
- **SAMSum Dataset:** Messenger-like conversations dataset created and annotated by linguists at Samsung R&D Institute Poland.
- **PyTorch:** For model training and optimization.
- **FastAPI:** For building and serving the API that handles real-time summarization requests and model training.
- **Uvicorn:** ASGI server to run the FastAPI application.

### SAMSum Dataset Details  
The SAMSum dataset was created by linguists to reflect everyday messenger-style conversations and is annotated for summarization purposes. It consists of 16k conversations with varying topics, informal to formal language, and various communication features like typos and emoticons.  
Dataset License: Non-commercial license (CC BY-NC-ND 4.0).

---

## **Results**

### MLOps Pipeline  
- **Best Model:** Linear Regression (R² = 0.88).  
- **Scalable Deployment:** Achieved through CI/CD and AWS integration.

### Generative AI Application  
- **Blog Generation:** Successfully deployed a scalable blog creation system using fine-tuned Hugging Face models and AWS services.

### Summarization Model  
- **Fine-tuned Performance:** Enhanced performance on generating summaries for informal, semi-formal, and formal messenger-like conversations, making it suitable for real-world applications.

---


