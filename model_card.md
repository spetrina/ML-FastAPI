# Model Card

## Model Details
The model is a RandomForestClassifier trained on U.S. census data to predict income categories (>50K or ≤50K). It leverages one-hot encoding for categorical variables and label binarization for the target. The model was trained using scikit-learn’s default parameters with a fixed random seed for reproducibility.

## Intended Use
This model is designed for income classification based on demographic and employment data. It is intended for educational and experimental purposes and for users familiar with ML pipelines processing structured tabular data. It is not recommended for use in high-stakes or production scenarios without further validation.

## Training Data
The training data consists of a processed subset of the census.csv dataset. It includes features such as workclass, education, marital status, occupation, race, and others. The dataset was split into training and testing, with categorical features handled through one-hot encoding.

## Evaluation Data
Evaluation was performed on a held-out test set comprising 20% of the original dataset to assess generalization. The same preprocessing steps used on training data were applied. Model metrics such as precision, recall, and F1 were calculated to quantify performance.

## Metrics
The model achieved the following metrics on the test dataset:
- Precision: 0.7419  
- Recall: 0.6384  
- F1 Score: 0.6863  

Additionally, performance metrics were computed on slices of data based on categorical feature values to analyze behavior across subgroups.

## Ethical Considerations
The model was trained on census data that may contain inherent societal biases. Performance disparities across subpopulations should be considered before deployment. This documentation highlights the need for cautious interpretation and potential bias audits for real-world applications.

## Caveats and Recommendations
- Model performance varies between subgroups; usage in critical applications should be preceded by further fairness testing and mitigation.  
- This implementation serves as a demonstration; production use requires additional robustness and security considerations.  
- Continuous monitoring and updating of the model are advisable as data distributions and social factors evolve.

---

This model card enhances transparency on what the model does, how well it performs, and ethical aspects to consider, guiding responsible use.