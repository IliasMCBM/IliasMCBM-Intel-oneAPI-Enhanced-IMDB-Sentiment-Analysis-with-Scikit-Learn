<p align="center">
  <img src="https://github.com/user-attachments/assets/1a290a9e-66cf-406f-8c76-f86b4fc01be9" alt="Intel oneAPI Enhanced IMDB Sentiment Analysis" width="500" style="background-color: rgba(255, 255, 255, 0.7);">
</p>


### Overview

This repository demonstrates the use of Intel's oneAPI technology to accelerate classic machine learning algorithms in Scikit-Learn for sentiment analysis tasks. By leveraging Intel’s advanced optimizations, we can efficiently run computationally demanding algorithms like Support Vector Classification (SVC), which would otherwise take significantly longer to execute on standard hardware.

### Project Purpose

The goal of this project is to showcase how Intel’s oneAPI can enhance the performance of machine learning models. For example, training and evaluating models like SVC, which is known for being computationally intensive, can be significantly faster using Intel's optimizations, even when working with large datasets. This speed-up enables data scientists and machine learning practitioners to iterate more quickly and efficiently.

### Key Benefits of Intel oneAPI

With Intel oneAPI, we utilize Intel hardware optimizations to boost the speed and efficiency of machine learning tasks, allowing for:

- **Faster Model Training:** By using Intel oneAPI, the training time for models like SVC is greatly reduced, which is particularly beneficial when working with large datasets or when rapid experimentation is needed.
- **Efficient Use of Resources:** Intel's optimizations ensure that machine learning workloads make the most of the available hardware, leading to better performance without the need for expensive infrastructure upgrades.

### Performance Improvement Visualization

Below is a graph that illustrates the performance improvements achieved when using Intel oneAPI with various Scikit-Learn algorithms. This visualization clearly shows the efficiency gains in terms of computation time and resource usage:

![image](https://github.com/user-attachments/assets/802497c2-7408-425c-a205-d2f2a0b6316b)

### Getting Started

#### Prerequisites

- **Docker:** Ensure Docker is installed and running on your machine. Docker allows you to containerize the application, ensuring consistent environments and dependencies.

#### Installation Steps
To run the project using Docker, follow these steps:

1. **Build the Docker Image**  
   Open your terminal or command prompt, navigate to the project directory, and build the Docker image using the following command:

   ```bash
   docker build -t imdb_predictor . 

2. **Run the Docker Container**
Once the image is built, run the container to execute the main.py script:

```bash
docker run imdb_predictor
```
### Repository Structure

- **IMDB_Classification.ipynb:** This Jupyter Notebook contains the code for preprocessing, model training, and evaluation. It provides a detailed walkthrough of how Intel oneAPI enhances model performance.
- **main.py:** A standalone script that executes the sentiment analysis task, utilizing Intel oneAPI optimizations for improved performance.
- **Dockerfile:** A Dockerfile that sets up a containerized environment for running `main.py`. This ensures the project can be easily executed without worrying about dependencies, making it ideal for sharing and deploying the application.
Rep

