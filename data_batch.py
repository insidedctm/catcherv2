import pandas as pd
from tensorpack.dataflow import DataFromGenerator, BatchData, DataFlow

class MyDataFlow(DataFlow):
  def __iter__(self):
    # load data from somewhere with Python, and yield them
    manifest = pd.read_csv("/Users/robineast/projects/catcher/labels.csv")
    exploded_manifest = manifest.apply(lambda row: list(map(lambda i: (row['filename'], i, 0), range(row['nofall_frame_start'],row['nofall_frame_end']+1))), axis=1)
    frame_and_labels = exploded_manifest.explode()

    for item in frame_and_labels:
      yield [(item[0], item[1]), item[2]]

df = MyDataFlow()
df.reset_state()

for datapoint in df:
    print(datapoint[0], datapoint[1])
