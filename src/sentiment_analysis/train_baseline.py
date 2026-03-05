import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sentiment_analysis.preprocessing import download_phrasebank, load_and_split_data
from sklearn.model_selection import GridSearchCV

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("sentiment_analysis")
csv_path = download_phrasebank()

def train_and_tune_baseline():
    train, val, test = load_and_split_data(csv_path)

    with mlflow.start_run(run_name="logreg_grid_search_tuned"):
        # vectorize
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X_train = vectorizer.fit_transform(train['text_clean'])
        X_val = vectorizer.fit_transform(val['text_clean'])

        param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
        grid_search = GridSearchCV(
                    LogisticRegression(max_iter=1000, class_weight='balanced'),
                    param_grid,
                    cv=5,
                    scoring='f1_weighted'
                    )
        grid_search.fit(X_train, train['sentiment'])
        model = grid_search.best_estimator_

        # log parameters
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("max_features", 5000)
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_param("best_C", grid_search.best_params_['C'])

        mlflow.log_metric("cv_f1", grid_search.best_score_) 

        # evaluate
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_pred, val['sentiment'])
        f1 = f1_score(y_pred, val['sentiment'], average='weighted')

        # log metrics
        mlflow.log_metric("val_accuracy", accuracy)
        mlflow.log_metric("val_f1", f1)

        # log model -- removed because it was causing errors
        # mlflow.sklearn.log_model(model, "model")

        print(f"Best C: {grid_search.best_params_['C']}")
        print(f"CV F1: {grid_search.best_score_:.3f}")
        print(f"Val Accuracy: {accuracy:.3f}, Val F1: {f1:.3f}")
        print("\nClassification Report:")
        print(classification_report(val['sentiment'], y_pred))

if __name__ == "__main__":
    train_and_tune_baseline()