import itertools

import torch
from torch.utils.data import IterableDataset, get_worker_info


class MyDataset(IterableDataset):
    def __init__(self, data,  batch_size, max_length):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.max_length = max_length

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # If no worker_info is provided, we are not using DataLoader workers, so yield all data
            iter_data = self.data
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.data, worker_id, None, num_workers)
            iter_data = list(iter_data)
        
        batch = []
        while True:   
            
            if len(iter_data) == 0:
                break

            
            start_pos = torch.randint( low = 0, high=len(iter_data)-self.max_length, size = (1,) )
            end_pos = start_pos+self.max_length 
            

            this_batch = iter_data[start_pos.item():end_pos.item()] 
            #this_batch = iter_data[0:self.max_length]
            batch.append(  this_batch )

            if len(batch) == self.batch_size:
                yield torch.tensor(batch, dtype=torch.int64)
                batch = []

        #if batch:
        #    yield self._format_batch(batch)

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask}

class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # If no worker_info is provided, we are not using DataLoader workers, so yield all data
            iter_data = iter(self.data)
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.data, worker_id, None, num_workers)

        batch = []
        for example in iter_data:
            tokenized_example = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            batch.append(tokenized_example)

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask}


if __name__ ==  "__main__":

    batch_size = 8
    data = torch.load("../data/training.pt")
    print( "data loaded...")
    
    dataset = MyDataset(data,  batch_size=batch_size, max_length=512)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=2)

    idx=0
    for batch in dataloader:
        
        print(batch[0:10])
        idx+=1
        if idx==100:
            break

