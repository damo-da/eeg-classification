from moabb1.datasets import Shin2017A
from utils import extract_epochs

def main():
    subjects = [1]

    dataset = Shin2017A()

    print('Loading data')
    all_data = dataset.get_data(subjects)

    print('Extracting epochs')
    data = extract_epochs(all_data, subjects)

    data[1].plot(show=True)

    print('Applying signal filtering')


if __name__ == '__main__':
    main()
