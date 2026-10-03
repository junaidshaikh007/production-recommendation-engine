# Day 7: Model Serving & Dashboard

## Objective

Now that we have our models evaluated and the Hybrid recommender performing best, Day 7 focuses on deploying the model. We will expose the recommender via a FastAPI application and create a simple web dashboard to interact with the system.

## 4-Task Plan

**Task 1: Planning and Model Persistence**
* Write this plan document.
* Add model persistence utilities (e.g., saving and loading the Hybrid model using `joblib` or `pickle`) so the API can load a pre-trained model.

**Task 2: FastAPI Application**
* Create `src/api/app.py` defining the FastAPI endpoints:
    * `GET /health`: For health checks.
    * `GET /recommend/{user_id}`: To get Top-K recommendations for a given user.
* Include dependency loading and request/response models.

**Task 3: Simple Dashboard**
* Create a simple HTML/JS frontend that allows a user to input their `user_id` and view recommendations.
* Serve the static dashboard directly from the FastAPI application.

**Task 4: Wrap-up & README Update**
* Update `README.md` to reflect the completion of the project (Day 7).
* Add instructions on how to start the FastAPI server and view the dashboard.
