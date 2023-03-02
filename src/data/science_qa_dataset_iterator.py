from src.data.science_qa_dataset_std import ScienceQADatasetStd


class ScienceQADatasetIterator:

    def __init__(self, dataset: ScienceQADatasetStd, batch_size: int = 100):
        self._dataset = dataset
        self.batch_size = batch_size if batch_size else len(self._dataset)
        self.num_batches = int(len(self._dataset) / self.batch_size)
        if len(self._dataset) % self.batch_size:
            self.num_batches += 1

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < self.num_batches:
            items = []
            for i in range(self.batch_size):
                try:
                    index = (self._index * self.batch_size) + i
                    items.append(self._dataset.__getitem__(index))
                except IndexError:
                    break
            self._index += 1
            return items
        else:
            raise StopIteration