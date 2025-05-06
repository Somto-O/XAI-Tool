import shap
import lime
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer

def explain_with_shap(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train)

def explain_with_lime(model, X_train):
    explainer = LimeTabularExplainer(X_train, training_labels=None, mode='classification')
    exp = explainer.explain_instance(X_train[0], model.predict_proba)
    exp.as_pyplot_figure()
