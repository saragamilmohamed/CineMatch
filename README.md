

# 🎬 CineMatch - Personalized Movie Recommendation Dashboard

CineMatch is an interactive, AI-powered movie recommendation system that provides **personalized movie suggestions** and **similar item exploration** using collaborative filtering models. Built with [Dash by Plotly](https://dash.plotly.com/) and deployed on [Hugging Face Spaces](https://huggingface.co/spaces), the project demonstrates a complete recommendation system pipeline — from training to web-based deployment.

<br>

## 🔍 Project Overview

This system is designed to:
- Predict a **user’s top-N personalized movie recommendations**.
- Display a **user’s movie rating history**.
- Present **item-based similarity recommendations**.
- Serve a clean, interactive dashboard for both users and movie exploration.

> Dataset used: [MovieLens Latest Small](https://grouplens.org/datasets/movielens/) (100k+ ratings by 600+ users on 9,000+ movies)

<br>

## 🚀 Live Demo

Check out the live demo on Hugging Face Spaces:  
🔗 [CineMatch on Hugging Face](https://huggingface.co/spaces/ali2yman/aliayman-dashDemo) 


<br>

## 🧠 Recommender System Details

### ✅ Model
We used a **Singular Value Decomposition (SVD)** model trained with `surprise` library:

- Trained on the MovieLens rating data.
- Serialized with `pickle` for fast inference.



### ✅ Features

#### 📄 User Page

* **Select User** from dropdown.
* View **User Rating History**.
* Input **N** for number of recommendations.
* View **Top-N Recommendations** with pagination.

#### 🎥 Item Page

* **Select Movie** from dropdown.
* View **Movie Metadata** (title, genres).
* View **Top-N Similar Movies** based on cosine similarity of latent features.
* Navigate through results with pagination.

<br>

## 🧰 Tech Stack

| Component     | Technology                    |
| ------------- | ----------------------------- |
| Data Source   | MovieLens Latest Small        |
| Modeling      | `scikit-surprise` (SVD Model) |
| Web Framework | Dash by Plotly                |
| Deployment    | Hugging Face Spaces           |
| Serialization | `pickle`                      |

<br>

## 📁 Project Structure

```
CineMatch/
│
├── app.py                  # Main Dash app script
├── model.pkl               # Trained SVD model
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── data/                   # MovieLens dataset files (ratings, movies, etc.)
```

<br>


---



## Team Credits

This project was collaboratively prepared by:

- Ali Ayman: [@ali2yman ](https://github.com/ali2yman)
- Amr Alaa: [@Amrokahla](https://github.com/Amrokahla)
- Mohamed Salama: [@mohamedsalama677](https://github.com/mohamedelsharkawy-coder)
- Sara Gamil: [@saragamilmohamed](https://github.com/mohamedsalama677)

Special thanks to the instructors of **ITI AI Pro Intake 45** for guidance and support.

---

## 📜 License

This project uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/), provided under the terms outlined by the GroupLens research group at the University of Minnesota.
CineMatch is an open-source academic project and should not be used for commercial purposes.



