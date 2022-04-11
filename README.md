# TaskIO (TIO)

A minimal framework for handling tasks input and output processing. This is
heavily inspired by [Google's SeqIO](https://github.com/google/seqio) but not
written with `tf.data`. For the time being, this
uses [HuggingFace's Dataset](https://huggingface.co/docs/datasets/) framework as
the backbone.

### Install

To install run:

```shell
git clone https://github.com/gabeorlanski/taskio.git
cd taskio
pip install -r requirements.txt
pip install -e .
```

### Basic Guide

Each [`Task`](https://github.com/gabeorlanski/taskio/blob/f7ed6594fb73f74489d2b700c05e8c758b4f6ff3/tio/task.py)
has 4 key elements that make it up:

1. A `SPLIT_MAPPING` that maps a split name (e.g. `train`,`validation`) to some
   key value.
2. A `tokenizer` for automatically encoding and decoding the inputs
3. Two list of callable functions `preprocessors` and `postprocessors` that are
   for preprocessing and postprocessing respectively. Each callable in these
   must take in a single dictionary argument. (More advanced things can be done
   with
   [`functools.partial`](https://docs.python.org/3/library/functools.html#functools.partial))
4. A set of `metric_fns` that are a list of callables. Each function must have
   the signature `predictions: List[str], targets: List[str]`

To create your own task, you must first subclass the `Task` class:

```python
from tio import Task


@Task.register('example')
class ExampleTask(Task):
    SPLIT_MAPPING = {
        "train"     : "path to the train file",
        "validation": "Path to the validation file"
    }

    @staticmethod
    def map_to_standard_entries(sample: Dict) -> Dict:
        sample['input_sequence'] = sample['input']
        sample['target'] = sample['output']
        return sample

    def dataset_load_fn(self, split: str) -> Dataset:
        # This is only an example and will not work
        return Dataset.load_dataset(self.SPLIT_MAPPING[split])
```

The first step is to register your task in the `Task` registry (Inspired by
AllenNLP's registrable). Then you must set the `SPLIT_MAPPING` and override the
two functions:

1. `map_to_standard_entries`: When preprocessing and postprocessing, the `Task`
   class expects there to be two columns `input_sequence` and `target`. This
   function maps the input to those columns.
2. `dataset_load_fn`: Function to load the dataset.

To actually use the task and get the dataset use:

```python
from tio import Task

task = Task.get_task(
    name='example',
    tokenizer=tokenizer,
    preprocessors=preprocessors,
    postprocessors=postprocessors,
    metric_fns=metric_fns
)
tokenized_dataset = task.get_split("train")

...

metrics = task.evaluate(
    **task.postprocess_raw_tokens(predictions, tokenized_dataset['labels'])
)
```

TODO: Make this less clunky