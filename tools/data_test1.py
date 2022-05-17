class nolabel_iter:

    def __init__(self, num):
        self.name = "nolabel"
        self.num = num
        self.count = -1
        self.res = list(range(num))

    def __iter__(self):
        for i in range(self.num):
            yield self.name + str(i)

    def __len__(self):
        return self.num


class label_iter:

    def __init__(self, num):
        self.name = 'label  '
        self.num = num
        self.count = -1
        self.res = list(range(num))

    def __iter__(self):
        for i in range(self.num):
            yield self.name + str(i)

    def __len__(self):
        return self.num


if __name__ == '__main__':
    train_nolabel_loader = nolabel_iter(29)
    train_label_loader = label_iter(7)

    num_batches = len(train_label_loader) + len(train_nolabel_loader)
    label_nolabel_ratio = len(train_nolabel_loader) // len(train_label_loader)
    rest_nolabel_num = len(train_nolabel_loader) % len(train_label_loader)
    rest_nolabel_id = num_batches - rest_nolabel_num + 1

    for i in range(2):
        train_label_iter = iter(train_label_loader)
        train_nolabel_iter = iter(train_nolabel_loader)
        print(f'EPOCH: {i}')
        for batch_id in range(1, num_batches + 1):
            if batch_id >= rest_nolabel_id:
                batch = next(train_nolabel_iter)
            else:
                if batch_id % (label_nolabel_ratio + 1) == 0:
                    batch = next(train_label_iter)
                else:
                    batch = next(train_nolabel_iter)
            print(batch)
