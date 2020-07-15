
def add_dataset_info(model, dataset):
    """
    Adds dataset info fields into model
    """
    model.input_interval = dataset.input_interval
    model.prediction_interval = dataset.prediction_interval
    model.categories = dataset.categories
    model.category_count = len(dataset.categories) + 1
    