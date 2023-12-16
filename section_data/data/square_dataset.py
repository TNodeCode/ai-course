from torch.utils.data import Dataset
import pandas as pd

class CSVDataset(Dataset):
    """
    This class loads data from a CSV file
    """
    def __init__(self, filename="./square.csv"):
        """
        Initialize the dataset class
        :param filename: The filename of the CSV file
        """
        self.data = pd.read_csv(filename)

    def __len__(self):
        """
        This function returns the total number of items in the dataset.
        We are using a pandas data frame in this dataset which has an attribut named shape.
        The first dimension of shape is equal to the number of items in the dataset.
        :return: The number of rows in the CSV file
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        This function returns a single tuple from the dataset.
        :param idx: The index of the tuple that should be returned.
        :return: Tuple of an x-value and a y-value
        """
        return self.data.iloc[idx]["x"], self.data.iloc[idx]["y"]