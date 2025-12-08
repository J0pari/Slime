import struct

# Read MNIST images
with open('../data/train-images-idx3-ubyte', 'rb') as f:
    magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
    images = f.read()

# Read MNIST labels
with open('../data/train-labels-idx1-ubyte', 'rb') as f:
    magic, num_labels = struct.unpack('>II', f.read(8))
    labels = f.read()

# Write binary files
with open('../data/mnist_images.bin', 'wb') as f:
    f.write(images)

with open('../data/mnist_labels.bin', 'wb') as f:
    f.write(labels)

print(f'Generated mnist_images.bin: {len(images)} bytes ({num} images)')
print(f'Generated mnist_labels.bin: {len(labels)} bytes ({num_labels} labels)')