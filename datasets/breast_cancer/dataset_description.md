The Breast Cancer Wisconsin dataset is a classic machine learning dataset for binary classification
of breast cancer diagnosis. This dataset contains features computed from a digitized image of a
fine needle aspirate (FNA) of a breast mass, describing characteristics of the cell nuclei present
in the image.

The dataset includes 30 features for each sample, computed from the cell nuclei characteristics:

- 10 features for mean values (radius, texture, perimeter, area, smoothness, compactness,
  concavity, concave points, symmetry, fractal dimension)
- 10 features for standard error values of the same characteristics
- 10 features for worst values of the same characteristics

The target variable 'class' is binary:

- M: Malignant (cancerous)
- B: Benign (non-cancerous)

This dataset is commonly used for binary classification tasks in medical diagnosis, with the goal
of predicting whether a breast mass is malignant or benign based on the computed cell nuclei features.
