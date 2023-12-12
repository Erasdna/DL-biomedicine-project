from torch.utils.data import DataLoader

from datasets.dataset import *


class CustomSimpleDataset(FewShotDataset):
    _dataset_name = 'custom'
    _data_dir = './data'
    
    def __init__(self, batch_size, root='./data/', mode='train', min_samples=20):
        if mode == 'train':
            self.samples = torch.randn((200, 100), generator=torch.Generator().manual_seed(1))
            self.labels = torch.randint(0, 5, (200, ), generator=torch.Generator().manual_seed(1))
        elif mode == 'val':
            self.samples = torch.rand((50, 100), generator=torch.Generator().manual_seed(2))
            self.labels = torch.randint(0, 5, (50, ), generator=torch.Generator().manual_seed(2))
        else:
            self.samples = torch.rand((50, 100), generator=torch.Generator().manual_seed(3))
            self.labels = torch.randint(0, 5, (50, ), generator=torch.Generator().manual_seed(3))
        self.batch_size = batch_size
        super().__init__()

    def __getitem__(self, i):
        return self.samples[i], self.labels[i]

    def __len__(self):
        return len(self.samples)

    @property
    def dim(self):
        return self.samples.shape[1]

    def get_data_loader(self) -> DataLoader:
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(self, **data_loader_params)

        return data_loader


class CustomSetDataset(FewShotDataset):
    _dataset_name = 'custom'
    _data_dir = './data'

    def __init__(self, n_way, n_support, n_query, n_episode=100, root='./data', mode='train'):
        self.n_way = n_way
        self.n_episode = n_episode
        min_samples = n_support + n_query

        self.samples_all= torch.randn((200, 100), generator=torch.Generator().manual_seed(1))
        self.labels_all = torch.randint(0, 5, (200, ), generator=torch.Generator().manual_seed(1))


        self.categories = torch.unique(self.labels_all) # Unique annotations

        self.x_dim = self.samples_all.shape[1]

        self.sub_dataloader = []

        sub_data_loader_params = dict(batch_size=min_samples,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for annotation in self.categories:
            idxs = (self.labels_all == annotation)
            samples = self.samples_all[idxs]
            labels = self.labels_all[idxs]
            sub_dataset = SubDataset(samples, labels)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

        super().__init__()

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.categories)

    @property
    def dim(self):
        return self.x_dim

    def get_data_loader(self) -> DataLoader:
        sampler = EpisodicBatchSampler(len(self), self.n_way, self.n_episode)
        data_loader_params = dict(batch_sampler=sampler, num_workers=4, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(self, **data_loader_params)
        return data_loader

class SubDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, i):
        return self.samples[i], self.labels[i]
        

    def __len__(self):
        return len(self.samples)

    @property
    def dim(self):
        return PROTDIM

if __name__ == "__main__":
    d = CustomSetDataset(5, 5, 15)
