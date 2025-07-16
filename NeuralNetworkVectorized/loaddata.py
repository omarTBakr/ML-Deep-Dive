import pathlib 
from torchvision.datasets import MNIST
from torchvision import transforms


def download_mnist_data(data_dir: str):
    """
    Download MNIST dataset to the specified directory if it does not exist.
    """
    data_path = pathlib.Path(data_dir)
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        download = True
        transform = transforms.Compose([transforms.ToTensor()])
        MNIST(root=data_dir, train=True, download=download, transform=transform)
        MNIST(root=data_dir, train=False, download=download, transform=transform)

def load_mnist_data(data_dir='./data'):
    """
    Load MNIST dataset from the specified directory.
    
    Args:
        data_dir (str): Directory where the MNIST data is stored.
        train (bool): If True, load training data; if False, load test data.
    
    Returns:
        torchvision.datasets.MNIST: The MNIST dataset.
    """
    data_path = pathlib.Path(data_dir)
    if not data_path.exists():
        download_mnist_data(data_dir)
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root=data_dir, train=True, download=False, transform=transform)
    test_dataset = MNIST(root=data_dir, train=False, download=False, transform=transform)
    return train_dataset, test_dataset
