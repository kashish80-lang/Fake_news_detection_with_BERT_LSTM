# 📰 Fake News Detection using Machine Learning

This project focuses on detecting **fake and real news articles** using **Machine Learning and Natural Language Processing (NLP)** techniques.  
The implementation is done in a **Jupyter Notebook (.ipynb)** for easy understanding and experimentation.

---

## 🚀 Project Overview

With the rapid spread of misinformation on social media and news platforms, identifying fake news has become crucial.  
This system classifies news articles as **Fake** or **Real** based on their textual content.
Used LSTM AND BERT together to improve accuracy.
real time based using gradio and hugging face.

---

## ✨ Features
- Text preprocessing and cleaning
- Stopword removal and tokenization
- Feature extraction using **TF-IDF / Count Vectorizer**
- Machine Learning models:
  - Logistic Regression
  - Naive Bayes
  - Passive Aggressive Classifier
- Model evaluation using accuracy and confusion matrix

---

## 📂 Project Structure
Fake-News-Detection/

│── fake_news_detection.ipynb

│── dataset/

│ ├── Fake.csv

│ └── True.csv

│── README.md



---

## 📊 Dataset
- **Fake.csv** – Contains fake news articles (23481 rows)
- **True.csv** – Contains real news articles (21417 rows)

Dataset Source: Kaggle Fake News Dataset
link:- https://www.kaggle.com/datasets/jainpooja/fake-news-detection
---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib / Seaborn
- Jupyter Notebook

---

## ▶️ How to Run the Project
1. Clone the repository:

git clone https://github.com/your-username/Fake-News-Detection.git

2. Navigate to the project folder:

cd Fake-News-Detection


3. Install required libraries:

pip install pandas numpy scikit-learn nltk matplotlib seaborn


4. Open the notebook:

jupyter notebook fake_news_detection.ipynb


5. Run all cells to train and evaluate the model.

## 📈 Output
1. Classifies news as Fake or Real
2. Displays model accuracy and performance metrics
3. Confusion matrix for result analysis
4. Accuracy:93.02%

## 🎯 Future Enhancements
1. Multilingual fake news detection
2. use method to decrese space complexity on large datasets.

## 👤 Author

### Kashish Malviya
MCA – NIT Trichy
```bash
