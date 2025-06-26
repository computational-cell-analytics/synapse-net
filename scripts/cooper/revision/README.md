# Improving the AZ model

Scripts for improving the AZ annotations, training the AZ model, and evaluating it.

The most important scripts are:
- For improving and updating the AZ annotations:
    - `prediction.py`: Run prediction of vesicle and boundary model.
    - `thin_az_gt.py`: Thin the AZ annotations, so that it aligns only with the presynaptic membrane. This is done by intersecting the annotations with the presynaptic compartment, using predictions from the network used for compartment segmentation.
    - `assort_new_az_data.py`: Create a new version of the annotation, renaming the dataset, and creating a cropped version of the endbulb of held data.
    - `merge_az.py`: Merge AZ annotations with predictions from model v4, in order to remove some artifacts that resulted from AZ thinning.
- For evaluating the AZ predictions: 
    - `az_prediction.py`: Run prediction with the AZ model.
    - `run_az_evaluation.py`: Evaluate the predictions of an AZ model.
    - `evaluate_result.py`: Summarize the evaluation results.
- And for training: `train_az_gt.py`. So far, I have trained:
    - v3: Trained on the initial annotations.
    - v4: Trained on the thinned annotations.
    - v5: Trained on the thinned annotations with an additional distance loss (did not help).
    - v6: Trained on the merged annotations.
