# Safety Incident Reporting Dataset

This repository contains a spreadsheet **SIRE.xlsx** with records of safety events.

The relevant columns for machine learning are:

- **Tipo evento** – category label for the event
- **Descripción detallada del evento** – text description of what happened

The example script `train_transformer.py` demonstrates how to train a text
classification model using these columns.

## Getting started

1. Install Python packages:
   ```bash
   pip install pandas scikit-learn torch transformers
   ```
2. Run the training script:
   ```bash
   python train_transformer.py
   ```

The script loads `SIRE.xlsx`, splits the data into train/test sets, tokenizes the
text with a Spanish BERT model, and fine-tunes it to predict the event type.

This is a simple starting point; adjust hyperparameters and preprocessing as
needed for your application.
