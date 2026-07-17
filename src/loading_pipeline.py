# =====================================================
# Complete Loading Pipeline
# =====================================================

def loading_pipeline(
    self,
    save=False
):
    """
    Complete dataset loading pipeline.

    Parameters
    ----------
    save : bool, default=False
        Save the loaded dataset to data/processed.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    # Load Dataset
    self.load_dataset()

    if self.df is None:
        raise ValueError("Dataset could not be loaded.")

    # Dataset Overview
    self.dataset_info()

    self.dataset_shape()

    self.column_names()

    self.data_types()

    # Preview Dataset
    self.head()

    self.tail()

    # Data Quality
    self.missing_values()

    self.duplicate_values()

    self.unique_values()

    # Statistics
    self.numerical_columns()

    self.categorical_columns()

    self.memory_usage()

    self.statistical_summary()

    # Optional Save
    if save:
        self.save_dataset()

    print("\n" + "=" * 60)
    print("DATASET LOADING PIPELINE COMPLETED")
    print("=" * 60)

    return self.df
