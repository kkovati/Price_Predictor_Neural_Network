import copy


def add_dataset_info(model, dataset):
    """
    Adds dataset info fields into model
    """
    model.input_interval = dataset.input_interval
    model.prediction_interval = dataset.prediction_interval
    model.categories = copy.deepcopy(dataset.categories)
    model.categories.sort()
    model.category_count = len(dataset.categories) + 1
    