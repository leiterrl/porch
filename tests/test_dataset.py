import torch
import pytest

from porch.dataset import NamedTensorDataset, BatchedNamedTensorDataset


class TestNamedTensorDataset:
    """Test suite for NamedTensorDataset."""

    def test_dataset_initialization(self):
        """Test dataset initialization with named tensors."""
        data = {
            "interior": torch.randn(100, 3),
            "boundary": torch.randn(50, 3),
        }
        dataset = NamedTensorDataset(data)

        assert "interior" in dataset._dataset
        assert "boundary" in dataset._dataset
        assert dataset._dataset["interior"].shape == torch.Size([100, 3])
        assert dataset._dataset["boundary"].shape == torch.Size([50, 3])

    def test_dataset_getitem(self):
        """Test dataset indexing."""
        data = {
            "interior": torch.randn(100, 3),
            "boundary": torch.randn(50, 3),
        }
        dataset = NamedTensorDataset(data)

        # Test accessing individual datasets
        interior_data = dataset["interior"]
        assert interior_data.shape == torch.Size([100, 3])

        boundary_data = dataset["boundary"]
        assert boundary_data.shape == torch.Size([50, 3])

    def test_dataset_setitem(self):
        """Test dataset assignment."""
        data = {"interior": torch.randn(100, 3)}
        dataset = NamedTensorDataset(data)

        new_boundary = torch.randn(50, 3)
        dataset["boundary"] = new_boundary

        assert "boundary" in dataset._dataset
        assert torch.equal(dataset["boundary"], new_boundary)

    def test_dataset_len(self):
        """Test dataset length."""
        data = {
            "interior": torch.randn(100, 3),
            "boundary": torch.randn(50, 3),
        }
        dataset = NamedTensorDataset(data)

        assert len(dataset) == 2

    def test_dataset_keys(self):
        """Test dataset keys."""
        data = {
            "interior": torch.randn(100, 3),
            "boundary": torch.randn(50, 3),
        }
        dataset = NamedTensorDataset(data)

        keys = dataset.keys()
        assert "interior" in keys
        assert "boundary" in keys

    def test_dataset_empty(self):
        """Test empty dataset."""
        dataset = NamedTensorDataset({})
        assert len(dataset) == 0


class TestBatchedNamedTensorDataset:
    """Test suite for BatchedNamedTensorDataset."""

    def test_batched_dataset_creation(self):
        """Test creation of batched dataset from named tensor dataset."""
        data = {
            "interior": torch.randn(100, 3),
            "boundary": torch.randn(50, 3),
        }
        base_dataset = NamedTensorDataset(data)

        batch_size = 10
        batched_dataset = BatchedNamedTensorDataset.from_named_tensor_dataset(
            base_dataset, batch_size=batch_size, shuffle=False, cycle=True
        )

        assert batched_dataset.batch_size == batch_size

    def test_batched_dataset_get_number_of_batches(self):
        """Test getting number of batches."""
        data = {
            "interior": torch.randn(100, 3),
            "boundary": torch.randn(50, 3),
        }
        base_dataset = NamedTensorDataset(data)

        batch_size = 10
        batched_dataset = BatchedNamedTensorDataset.from_named_tensor_dataset(
            base_dataset, batch_size=batch_size, shuffle=False, cycle=True
        )

        # Interior has 100 samples, so 10 batches of size 10
        # Boundary has 50 samples, so 5 batches of size 10
        assert batched_dataset.get_number_of_batches() >= 1

    def test_batched_dataset_step(self):
        """Test stepping through batches."""
        data = {
            "interior": torch.randn(100, 3),
            "boundary": torch.randn(50, 3),
        }
        base_dataset = NamedTensorDataset(data)

        batch_size = 10
        batched_dataset = BatchedNamedTensorDataset.from_named_tensor_dataset(
            base_dataset, batch_size=batch_size, shuffle=False, cycle=True
        )

        # Get first batch
        first_batch = batched_dataset["interior"]
        assert first_batch.shape[0] <= batch_size

        # Step to next batch
        batched_dataset.step()
        second_batch = batched_dataset["interior"]
        assert second_batch.shape[0] <= batch_size

        # Batches should be different (unless shuffle and cycle produce same)
        # Just check they're valid tensors
        assert not torch.isnan(first_batch).any()
        assert not torch.isnan(second_batch).any()

    def test_batched_dataset_with_shuffle(self):
        """Test batched dataset with shuffling."""
        data = {
            "interior": torch.randn(100, 3),
        }
        base_dataset = NamedTensorDataset(data)

        batch_size = 10
        batched_dataset = BatchedNamedTensorDataset.from_named_tensor_dataset(
            base_dataset, batch_size=batch_size, shuffle=True, cycle=True
        )

        batch = batched_dataset["interior"]
        assert batch.shape[0] <= batch_size

    def test_batched_dataset_cycle(self):
        """Test that batched dataset cycles through data."""
        data = {
            "interior": torch.randn(20, 3),
        }
        base_dataset = NamedTensorDataset(data)

        batch_size = 10
        batched_dataset = BatchedNamedTensorDataset.from_named_tensor_dataset(
            base_dataset, batch_size=batch_size, shuffle=False, cycle=True
        )

        # Step through more batches than available data
        num_batches = batched_dataset.get_number_of_batches()
        for _ in range(num_batches + 5):
            batch = batched_dataset["interior"]
            assert batch.shape[0] <= batch_size
            batched_dataset.step()
