import numpy as np


class SequentialDataLoaderWrapper:

    def __init__(self, data_loaders):
        self.data_loaders = data_loaders

    def __iter__(self):
        for data_loader in self.data_loaders:
            for batch in data_loader:
                yield data_loader.name, *batch

    def __len__(self):
        return np.sum([len(data_loader) for data_loader in self.data_loaders])


class label_iter:

    def __init__(self, name, num):
        self.name = name
        self.num = num

    def __iter__(self):
        for i in range(self.num):
            yield self.name, i

    def __len__(self):
        return self.num


if __name__ == '__main__':
    train_label_loaders = []
    for i, num in enumerate(range(3, 5)):
        train_label_loaders.append(label_iter(name=f'dataset-{i}', num=num))

    train_label_loader = SequentialDataLoaderWrapper(train_label_loaders)
    num_batches = len(train_label_loader)

    for i in range(2):
        train_label_iter = iter(train_label_loader)
        print(f'EPOCH: {i}')
        for batch_id in range(1, num_batches + 1):
            try:
                data_name1, data_name2, batch = next(train_label_iter)
            except StopIteration:
                train_label_iter = iter(train_label_loader)
                data_name1, data_name2, batch = next(train_label_iter)
            print(f'{data_name1} {data_name2} {batch}')
