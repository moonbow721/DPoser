
class DataLoaderForIterations():
    '''
    This class tries to mix two things:
        - multi-current efficiency of DataLoader in pytorch
        - yielding samples in the number of iterations, not till DataLoader would end
    '''
    def __init__(self, dataloader, num_iterations=1):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.num_iterations = num_iterations
        self.cur_iter = 0
    
    def __len__(self):
        return self.num_iterations
        
    def __iter__(self):
        self.cur_iter = 0
        self.iterator = iter(self.dataloader)
        return self
    
    def __next__(self):
        if self.cur_iter < self.num_iterations: 
            try:
                x = self.iterator.next()
            except StopIteration:
                self.iterator = iter(self.dataloader)
                x = self.iterator.next()
            self.cur_iter += 1
            return x
        
        else:
            raise StopIteration


class MultiDataLoaderForIterations():
    '''
    This class combines properties of DataLoaderForIterations 
        with sampling from several dataloaders simultaneously.
    
    The resulting batch_size is fixed. 
    Data from different dataloaders stacked together in one batch.
    
    drop_last = True will be used, so be sure that length of dataset is enough to collect a batch.
    '''
    def __init__(self, dataloaders, num_iterations=1):
        ''' dataloaders : list of dataloaders (pytorch DataLoader class)
        '''
        self.dataloaders = dataloaders
        for i, dl in enumerate(self.dataloaders):
            assert len(dl.dataset) >= dl.batch_size, f'Dataset size of {i} dataset is less than batch_size!'
            
        self.iterators = [iter(dl) for dl in self.dataloaders]
        self.num_iterations = num_iterations
        self.cur_iter = 0

    def __len__(self):
        return self.num_iterations
        
    def __iter__(self):
        self.cur_iter = 0
        self.iterators = [iter(dl) for dl in self.dataloaders]
        return self
    
    def __next__(self):
        if self.cur_iter < self.num_iterations: ### stopping criterion
            out = []
            for i in range(len(self.iterators)):
                try:
                    x = self.iterators[i].next()
                except StopIteration:
                    self.iterators[i] = iter(self.dataloaders[i])
                    x = self.iterators[i].next()
                out.append(x)
            
            self.cur_iter += 1
            return out
        else:
            raise StopIteration



def main():

    import torch

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, n=1_000, add=0):
            self.n = n
            self.add = add
            self.db = torch.arange(n)

        def __len__(self):
            return len(self.db)

        def __getitem__(self, idx):
            return self.db[idx] + self.add



    ### test DataLoaderForIterations
    print('\n'+'#'*50+'\n\ntest DataLoaderForIterations\n\n'+'#'*50+'\n')
    num_samples = 16
    batch_size = 3
    shuffle = True
    num_workers = 0
    drop_last = True

    ds = CustomDataset(num_samples)

    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last) 

    num_iterations = 100
    dl = DataLoaderForIterations(dl, num_iterations)

    for i, x in enumerate(dl):
        print(f'{i:4} {[int(i) for i in x]}')

    
    ### test MultiDataLoaderForIterations
    print('\n'+'#'*50+'\n\ntest MultiDataLoaderForIterations\n\n'+'#'*50+'\n')
    lens = (5, 21, 9)
    batch_sizes = (2,4,8)
    adds = (0,100, 10000)

    ds = [CustomDataset(l,a) for l,a in zip(lens,adds)]
    dl = [torch.utils.data.DataLoader(dataset, batch_size=batch, num_workers=0, shuffle=True, drop_last=True) 
                                                    for batch,dataset in zip(batch_sizes,ds)]

    multidl = MultiDataLoaderForIterations(dl, num_iterations=10)

    for x,y,z in multidl:
        print(x)
        print(y)
        print(z)
        print()




if __name__ == '__main__':
    main()