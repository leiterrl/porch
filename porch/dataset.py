import math
import torch
from torch import Tensor


# class PorchDatasetBase:
#     def __init__(self, config: PorchConfig, geometry: Geometry) -> None:
#         self.config = config
#         self.geometry = geometry
#         logging.info("Init dataset...")

#     def generate_data(self):
#         bc_tensors = []
#         logging.info("Generating BC data...")
#         for bc in self.geometry.get_boundary_conditions():
#             bc_data = bc.get_data(device=self.config.device)
#             bc_tensors.append(bc_data)
#         boundary_data = torch.cat(bc_tensors)

#         logging.info("Generating interior data...")
#         interior_data = self.geometry.get_random_samples(device=self.config.device)
#         self.dataset = TensorDataset(boundary_data, interior_data)


class NamedTensorDataset:
    def __init__(self, tensor_dict: "dict[str, Tensor]") -> None:
        self._dataset = tensor_dict
        self._names = list(tensor_dict.keys())

    def __getitem__(self, name: str) -> Tensor:
        if not name in self._names:
            raise KeyError('Unknown name: {}'.format(name))
        return self._dataset[name]

class BatchedNamedTensorDataset(NamedTensorDataset):
    def __init__(self, tensor_dict: "dict[str, Tensor]", batch_size: int, shuffle: bool, cycle: bool) -> None:
        '''Dataset for batching based on dataloaders.
        Note: the term dataset is here in conflict with the pytorch notion of a dataset.
        Parameters:
            tensor_dict
                the datasets
            batch_size
                the (combined) size of a batch. It is more or less evenly split to the tensors in tensor_dict.
            shuffle: 
                whether to shuffle the batches in each new iteration
            cycle: 
                whether to cycle a dataloader when its end is reached.
                Note that in this case, the number of datapoints changes!
        '''
        assert batch_size > 0
        super().__init__(tensor_dict)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cycle = cycle
        self.dataloader_iters = dict()
        self.current_dataset = None

        ## generate dataloaders which do the batching and shuffling
        self.dataloader_dict = dict()
        total_size = sum(len(t) for t in tensor_dict.values())
        # spread data more or less evenly
        for key, dataset in tensor_dict.items():
            self.dataloader_dict[key] = torch.utils.data.DataLoader(
                dataset,
                batch_size=math.ceil(len(dataset) / total_size * batch_size),
                shuffle=shuffle,
            )
        # maximal number of batches
        dataloader_lens = tuple(len(dl) for dl in self.dataloader_dict.values())
        assert all(l > 0 for l in dataloader_lens), 'dataloaders should at least contain one element!'
        self.max_num_batches = max(dataloader_lens)
    
    @classmethod
    def from_named_tensor_dataset(cls, named_tensor_dataset: NamedTensorDataset, batch_size: int, shuffle: bool, cycle: bool):
        return cls(named_tensor_dataset._dataset, batch_size, shuffle, cycle)

    def __getitem__(self, name: str) -> Tensor:
        if not name in self._names:
            raise KeyError('Unknown name: {}'.format(name))
        if self.current_dataset is None:
            raise ValueError('Call "step" before "__getitem__".')
        return self.current_dataset[name]
    
    def step(self) -> None:
        self.current_dataset = dict()
        for name in self._names:
            try:
                self.current_dataset[name] = next(self.dataloader_iters[name])
            except StopIteration:
                if self.cycle:
                    self.reset_iters([name])
                    self.current_dataset[name] = next(self.dataloader_iters[name])
                else:
                    shape = list(self._dataset[name].shape)
                    shape[0] = 0
                    device = device=self._dataset[name].device
                    self.current_dataset[name] = torch.empty(shape, device=device)
            except KeyError:
                raise ValueError('Call "reset_iters" before "step".')
    
    def reset_iters(self, names: "list[str]" = None) -> None:
        if names is None:
            names = self._names
        for name in names:
            dataloader = self.dataloader_dict[name]
            self.dataloader_iters[name] = iter(dataloader)

    def get_number_of_batches(self) -> int:
        return self.max_num_batches
    
    def get_lengths(self) -> int:
        return {name: len(ds) for name, ds in self._dataset.items()}

    def get_current_lengths(self) -> int:
        return {name: len(ds) for name, ds in self.current_dataset.items()}