# 5. Keras and the High-Level Path

> **Stage 05 · DL Frameworks** — The same network with a tenth of the ceremony: `Sequential`, `compile`, `fit`, and when that abstraction helps rather than hides.
>
> **You'll be able to:** build, train and evaluate a model in Keras in one screen of code, and decide honestly whether your project should use Keras or PyTorch.
>
> **Prerequisites:** [3. Anatomy of a Training Loop](03-training-loops.md) · [4. CNNs and RNNs](../04-deep-learning/04-cnns-rnns.md)

## Why this exists

The PyTorch loop you just wrote is explicit on purpose: you can see every step, which is what you want when something is wrong. But most of the time nothing is wrong, and typing the same seven lines for the tenth time is friction that buys nothing.

Keras is the framework that says: declare the layers, call `compile` with a loss and an optimizer, call `fit` with data. Training, validation, metrics, progress bars, callbacks and checkpointing are one argument each. TensorFlow bundles Keras as `tf.keras`; Keras also runs on JAX and PyTorch backends, so the API you learn is not locked to one engine.

The honest framing: *Keras optimises for time-to-first-working-model; PyTorch optimises for control. Use Keras to find out whether an idea is worth the control.*

## The mental model

```text
 PyTorch (explicit)                  Keras (declarative)
 ──────────────────                  ───────────────────
 model = nn.Sequential(...)          model = Sequential([...])
 criterion = nn.CrossEntropyLoss()   model.compile(loss=..., optimizer=..., metrics=[...])
 optimizer = AdamW(...)
 for epoch in range(E):              history = model.fit(x, y,
     model.train()                       epochs=E, batch_size=32,
     for xb, yb in dl:                   validation_split=0.2,
         zero_grad / forward /           callbacks=[checkpoint, early_stop])
         backward / step
     model.eval() ...
```

Everything the loop did by hand is inside `fit`. The four things you give up: per-batch control flow, custom gradient manipulation, unusual losses that need intermediate tensors, and a debugger-friendly stack trace when something is wrong.

## Core concepts

### Three ways to define a model, and which to use

**Sequential** is a straight line of layers — enough for most tabular, image and simple text models. **Functional API** wires layers as a graph, so you can have two inputs, skip connections, or two outputs; use it the moment your model branches. **Subclassing** (`class M(keras.Model)`) gives you a Python `call` method and full control, at the cost of losing some graph introspection. Start Sequential, move to Functional, subclass only when forced.

### `compile` is where the training contract lives

`compile(optimizer=..., loss=..., metrics=[...])` binds three things. Note the difference from PyTorch: Keras losses can be given as strings (`"sparse_categorical_crossentropy"`) because the framework knows whether your labels are one-hot or integer. `metrics` are for *reporting* only — they never affect gradients. Mixing up the two is how people think they are optimising accuracy when they are optimising cross-entropy.

### Callbacks replace your loop's bookkeeping

A callback is an object with hooks like `on_epoch_end`. The useful built-ins: `ModelCheckpoint` (save the best epoch by a monitored metric), `EarlyStopping` (stop when a metric stops improving, and restore the best weights), `ReduceLROnPlateau` (drop the learning rate), `CSVLogger` (write metrics to a file). This is exactly the code you would otherwise write by hand, already tested.

### `fit` returns history, and that is your first diagnostic

`history.history` is a dict of lists, one entry per epoch. If train loss falls and validation loss rises, you are overfitting; if both are flat, the learning rate or the data is wrong. Plot it before you change anything.

### It is not either/or

You can use Keras layers inside a PyTorch project, or export a Keras model and run it elsewhere. What you should not do is pick a framework by habit. Pick by asking: do I need to touch the gradient, the loop, or the loss internals? If yes for any, PyTorch. If no, Keras will get you there faster.

## Code

A complete training run — model, compile, fit with checkpointing and early stopping, evaluation, prediction — in one screen. This is tabular classification, the case where Keras shines.

```python
from __future__ import annotations

import numpy as np
import keras
from keras import layers

rng = np.random.default_rng(0)
x = rng.normal(size=(2000, 20)).astype("float32")
# a learnable rule so the model has something to find
y = ((x[:, 0] + 0.5 * x[:, 1] - x[:, 2]) > 0).astype("int32")

model = keras.Sequential(
    [
        layers.Input(shape=(20,)),                 # (None, 20) batch of 20 features
        layers.Dense(64, activation="relu"),       # -> (None, 64)
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),       # -> (None, 32)
        layers.Dense(2, activation="softmax"),     # -> (None, 2) class probs
    ],
    name="tabular_clf",
)
model.summary()                                    # parameter count per layer

model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss="sparse_categorical_crossentropy",        # integer labels, not one-hot
    metrics=["accuracy"],                          # reporting only
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best.keras", monitor="val_loss", save_best_only=True
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    ),
    keras.callbacks.CSVLogger("metrics.csv"),      # one row per epoch
]

history = model.fit(
    x, y,
    epochs=30,
    batch_size=32,
    validation_split=0.2,                          # last 20% held out
    callbacks=callbacks,
    verbose=2,
)

print({k: round(v[-1], 4) for k, v in history.history.items()})
probs = model.predict(x[:4], verbose=0)            # (4, 2)
print(probs.argmax(axis=1))
```

When the model branches, switch to the Functional API. The graph is the model, which makes multi-input models readable.

```python
# two inputs: numeric features and a category id, concatenated after embedding
num_in = keras.Input(shape=(20,), name="numeric")          # (None, 20)
cat_in = keras.Input(shape=(1,), dtype="int32", name="category")
emb = layers.Embedding(input_dim=50, output_dim=8)(cat_in) # (None, 1, 8)
emb = layers.Flatten()(emb)                                # (None, 8)
merged = layers.Concatenate()([num_in, emb])               # (None, 28)
out = layers.Dense(2, activation="softmax")(merged)

wide_model = keras.Model(inputs=[num_in, cat_in], outputs=out)
wide_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
```

## Real use case

A logistics company needs to predict whether a shipment will be late, from about forty tabular features. They have one data analyst, no deep-learning experience, and a two-week deadline for a pilot.

They start with a gradient-boosted tree, which is the right first answer for tabular data, and it gets most of the way. To see whether a neural net adds anything, they need a fair comparison within a day — not a framework to learn. Keras gets them there: `Sequential` with three `Dense` layers, `compile`, `fit` with `EarlyStopping`, done in an afternoon. The result is comparable to the tree, so the tree ships.

The point is not that Keras won. It is that the experiment cost a day instead of a week, and the decision was made on evidence. When a later project needed a custom loss that depended on an intermediate layer's output, they moved to PyTorch without hesitation — because they knew exactly which abstraction they had outgrown.

## Common mistakes

- **Mistake:** using `accuracy` as the loss. — **Instead:** `metrics` are reported, not optimised; pick a real loss such as cross-entropy.
- **Mistake:** `softmax` at the output with `from_logits=True` losses. — **Instead:** either keep softmax and use the default loss, or drop softmax and set `from_logits=True`; never both.
- **Mistake:** no `EarlyStopping`, so `epochs=200` trains to the end and overfits. — **Instead:** add early stopping with `restore_best_weights=True`.
- **Mistake:** forgetting `validation_split` or a validation set. — **Instead:** always hold data out; a training curve alone tells you nothing about generalisation.
- **Mistake:** fighting Keras to do something the loop-level API would make trivial. — **Instead:** that is the signal to switch to PyTorch, not to subclass harder.

## Checkpoint

1. Reproduce the tabular model above and plot `history.history["loss"]` against `val_loss` for 30 epochs. Where does overfitting start?
2. Add a third numeric input branch and merge all three with `Concatenate`, then print `model.summary()` and verify the parameter count by hand.
3. Write a custom callback that prints the learning rate at the end of every epoch, and confirm it changes when you add `ReduceLROnPlateau`.

**Self-check:** Overfitting starts where `val_loss` stops falling while `loss` keeps falling; with 2000 rows and this much capacity that is usually within the first ten epochs, which is why early stopping is not optional here.

## Go deeper

- [3. Anatomy of a Training Loop](03-training-loops.md) — what `fit` is doing underneath.
- [6. Hugging Face Transformers and Datasets](06-hugging-face.md) — the high-level path for models you did not design.
- [9. Transfer Learning](../04-deep-learning/09-transfer-learning.md) — Keras's `applications` module is the fastest way to try this.
- [5. Loss Functions](../02-math-and-ml-basics/05-loss-functions.md) — choosing the loss you pass to `compile`.

---
[← Previous: GPU Memory and Checkpoints](04-gpu-checkpoints.md) · [Stage index](README.md) · [Next: Hugging Face Transformers and Datasets](06-hugging-face.md)
