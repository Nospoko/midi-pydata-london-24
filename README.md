# midi-pydata-london-24
Code snippets and scripts for Can machines play the piano? presentation at PyData London 2024
## MIDI data
### MIDI to dataframe
To load a midi file into a dataframe, we use:
```py
import fortepyan as ff

piece = ff.MidiPiece.from_file(path="data/piano.mid")
```

```shell
python midi_to_dataframe.py
```
### Maestro dataset
You can find maestro dataset with "notes" and "source" column at huggingface
```py
from datasets import load_dataset

dataset = load_dataset("roszcz/maestro-sustain-v2")
```

| Split       | Performances | Duration (hours)  | Notes (millions) |
|-------------|--------------|-------------------|------------------|
| Train       | 967          | 161.3             | 5.73             |
| Validation  | 137          | 19.4              | 0.64             |
| Test        | 178          | 20.5              | 0.76             |
| Total       | 1282         | 201.2             | 7.13             |


### Average notes per second
We can calculate average notes per second in maestro dataset by counting rows in dataframes created from
dataset and dividing them by total time.
```py
for record in dataset:
    total_notes += len(record["notes"]["pitch"])
    total_time += max(record["notes"]["start"]) - min(record["notes"]["start"])
```
By using "start" time we are calculating how many notes were pressed in a second on average.

```shell
python notes_per_second.py
```

### Code Style

This repository uses pre-commit hooks with forced python formatting ([black](https://github.com/psf/black),
[flake8](https://flake8.pycqa.org/en/latest/), and [isort](https://pycqa.github.io/isort/)):

```sh
pip install pre-commit
pre-commit install
```

Whenever you execute `git commit` the files altered / added within the commit will be checked and corrected.
`black` and `isort` can modify files locally - if that happens you have to `git add` them again.
You might also be prompted to introduce some fixes manually.

To run the hooks against all files without running `git commit`:

```sh
pre-commit run --all-files
```
