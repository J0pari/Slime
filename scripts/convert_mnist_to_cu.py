import struct

with open('../data/train-images-idx3-ubyte', 'rb') as f:
    magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
    images = f.read()

with open('../data/train-labels-idx1-ubyte', 'rb') as f:
    magic, num_labels = struct.unpack('>II', f.read(8))
    labels = f.read()

with open('../slime/data/mnist_data.cu', 'w') as out:
    out.write('#ifndef MNIST_DATA_CU\n#define MNIST_DATA_CU\n\n')
    out.write(f'constexpr int MNIST_NUM_TRAIN = {num};\n')
    out.write(f'constexpr int MNIST_IMG_SIZE = {rows * cols};\n\n')

    out.write('static const unsigned char MNIST_IMAGES_HOST[] = {\n')
    for i in range(0, len(images), 16):
        out.write('    ' + ','.join(f'0x{b:02x}' for b in images[i:i+16]) + ',\n')
    out.write('};\n\n')

    out.write('static const unsigned char MNIST_LABELS_HOST[] = {\n')
    for i in range(0, len(labels), 16):
        out.write('    ' + ','.join(f'0x{b:02x}' for b in labels[i:i+16]) + ',\n')
    out.write('};\n\n')

    out.write('#endif\n')

print(f'Generated mnist_data.cu: {num} images, {len(images)} bytes')