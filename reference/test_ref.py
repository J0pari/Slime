import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import time

CHANNEL_N = 19
BATCH_SIZE = 16
POOL_SIZE = BATCH_SIZE * 10
CELL_FIRE_RATE = 0.5

class CAModel(tf.keras.Model):
    def __init__(self, channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.perceive = tf.keras.Sequential([Conv2D(80, 3, activation=tf.nn.relu, padding="SAME")])
        self.dmodel = tf.keras.Sequential([
            Conv2D(80, 1, activation=tf.nn.relu),
            Conv2D(self.channel_n, 1, activation=None, kernel_initializer=tf.zeros_initializer),
        ])
        self(tf.zeros([1, 3, 3, channel_n + 1]))

    @tf.function
    def call(self, x, fire_rate=None):
        gray, state = tf.split(x, [1, self.channel_n], -1)
        ds = self.dmodel(self.perceive(x))
        ds += tf.random.normal(tf.shape(ds), 0., 0.02)
        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
        living_mask = gray > 0.1
        residual_mask = update_mask & living_mask
        ds *= tf.cast(residual_mask, tf.float32)
        state += ds
        return tf.concat([gray, state], -1)

    @tf.function
    def initialize(self, images):
        state = tf.zeros([tf.shape(images)[0], 28, 28, self.channel_n])
        images = tf.reshape(images, [-1, 28, 28, 1])
        return tf.concat([images, state], -1)

    @tf.function
    def classify(self, x):
        return x[:,:,:,-10:]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = (x_train / 255.0).astype(np.float32)

def to_ten_dim_label(x, y):
    y_res = np.zeros(list(x.shape) + [10])
    y_expanded = np.broadcast_to(y, x.T.shape).T
    y_res[x >= 0.1, y_expanded[x >= 0.1]] = 1.0
    return y_res.astype(np.float32)

y_train_pic = to_ten_dim_label(x_train, y_train)

class SamplePool:
    def __init__(self, *, _parent=None, _parent_idx=None, **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = slots.keys()
        self._size = None
        for k, v in slots.items():
            if self._size is None: self._size = len(v)
            setattr(self, k, np.asarray(v))
    def sample(self, n):
        idx = np.random.choice(self._size, n, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        return SamplePool(**batch, _parent=self, _parent_idx=idx)
    def commit(self):
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)

ca = CAModel()

def batch_l2_loss(x, y):
    t = y - ca.classify(x)
    return tf.reduce_mean(tf.reduce_sum(t**2, [1, 2, 3]) / 2)

lr = 1e-3
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay([30000, 70000], [lr, lr*0.1, lr*0.01])
trainer = tf.keras.optimizers.Adam(lr_sched)

starting_indexes = np.random.randint(0, x_train.shape[0]-1, size=POOL_SIZE)
pool = SamplePool(x=ca.initialize(x_train[starting_indexes]).numpy(), y=y_train_pic[starting_indexes])

@tf.function
def train_step(x, y):
    with tf.GradientTape() as g:
        for i in tf.range(20):
            x = ca(x)
        loss = batch_l2_loss(x, y)
    grads = g.gradient(loss, ca.weights)
    grads = [g/(tf.norm(g)+1e-8) for g in grads]
    trainer.apply_gradients(zip(grads, ca.weights))
    return x, loss

print("Training reference model...", flush=True)
t0 = time.time()
for i in range(1, 5001):
    batch = pool.sample(BATCH_SIZE)
    x0 = np.copy(batch.x)
    y0 = batch.y
    q_bs = BATCH_SIZE // 4

    new_idx = np.random.randint(0, x_train.shape[0]-1, size=q_bs)
    x0[:q_bs] = ca.initialize(x_train[new_idx])
    y0[:q_bs] = y_train_pic[new_idx]

    new_idx = np.random.randint(0, x_train.shape[0]-1, size=q_bs)
    new_x, new_y = x_train[new_idx], y_train_pic[new_idx]
    new_x_r = tf.reshape(new_x, [q_bs, 28, 28, 1])
    mutate_mask = tf.cast(new_x_r > 0.1, tf.float32)
    mutated_x = tf.concat([new_x_r, x0[-q_bs:,:,:,1:] * mutate_mask], -1)
    x0[-q_bs:] = mutated_x
    y0[-q_bs:] = new_y

    x, loss = train_step(x0, y0)

    batch.x[:] = x
    batch.y[:] = y0
    batch.commit()

    if i <= 5 or i % 100 == 0:
        elapsed = time.time() - t0
        print(f"iter {i:5d} | loss {loss.numpy():.3f} | log10={np.log10(loss.numpy()):.3f} | {elapsed:.0f}s", flush=True)
