from train_conv3d import get_epoch_from_checkpoint_path
from Conv3DFallClassifier import FifoClipBuffer

def test_epoch_from_checkpoint_path():
    assert get_epoch_from_checkpoint_path('CatcherConv3D_Test2_6.ckpt') == 6, "Should be 6"
    assert get_epoch_from_checkpoint_path('CatcherConv3D_Test2_16.ckpt') == 16, "Should be 16"

def test_fifo_buffer():
    fifo = FifoClipBuffer(buffer_size=3)
    assert fifo.len() == 0, "fifo length should be 0 at start"
    fifo.add([0,0,0,0])
    assert fifo.len() == 1, "fifo length should be 1 after adding element"
    assert fifo.buffer == {0: [0,0,0,0]}, "dict should contain just 0s"
    fifo.add([1,1,1,1])
    assert fifo.len() == 2, "fifo length should be 2"
    assert fifo.buffer == {0: [0,0,0,0], 1: [1,1,1,1]}, "dict should contain just 0s and 1s"
    fifo.add([2,2,2,2])
    assert fifo.len() == 3, "fifo length should be 3 after adding element"
    assert fifo.buffer == {0: [0,0,0,0], 1: [1,1,1,1], 2: [2,2,2,2]}, "dict should contain just 0s,1s and 2s"
    fifo.add([3,3,3,3])
    assert fifo.len() == 3, "fifo length should be 3"
    assert fifo.buffer == {1: [1,1,1,1], 2: [2,2,2,2], 3: [3,3,3,3]}, "dict should contain just 1s, 2s and 3s"
    fifo.add([4,4,4,4])
    assert fifo.len() == 3, "fifo length should be 3"
    assert fifo.buffer == {4: [4,4,4,4], 2: [2,2,2,2], 3: [3,3,3,3]}, "dict should contain just 2s, 3s and 4s"
    assert min(fifo.buffer.keys()) == 2, "minimum index key should be 2"

if __name__ == "__main__":
    test_epoch_from_checkpoint_path()
    test_fifo_buffer()
    print("Everything passed")
