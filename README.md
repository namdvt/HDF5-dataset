## Create hdf5 dataset with unfixed size
```python
# initial unfixed size dataset
h5_train_data = hdf5.create_dataset('train/data', (size, 33075), maxshape=(None, 33075))
h5_train_label = hdf5.create_dataset('train/label', (size, 1), maxshape=(None, 1))

# append data
train_idx = 0
label = 0
for clazz in total_clazz:
    for data in os.listdir(clazz):
        h5_train_data[train_idx] = data
        h5_train_label = label
        train_idx += 1
    label += 1    

# size trimming
h5_train_data.resize((train_idx - 1, 33075))
h5_train_label.resize((train_idx - 1, 1))
```
## Load hdf5 dataset
```python
from hdf5dataset import Hdf5Dataset

train_dataset = Hdf5Dataset(hdf5file_address, 'train')
train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,
                                                drop_last=True)
for data, target in train_dataset:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()