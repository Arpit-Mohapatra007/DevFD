# DevFD – Future of Deepfake Detection Model

> A continual-learning deepfake detector that learns **new fake types one after another** without forgetting the old ones – our re-implementation of **DevFD**.

**Paper we took inspiration from:** *DevFD: Developmental Face Forgery Detection by Learning Shared and Orthogonal LoRA Subspaces* – <https://arxiv.org/pdf/2509.19230>

**Team:** Saish Kanade · Arpit Mohapatra · Harsh Vardhan Singh

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Core Idea Behind DevFD](#2-core-idea-behind-devfd)
3. [First Implementation](#3-first-implementation)
4. [Second Implementation](#4-second-implementation--finding-the-real-problem)
5. [Comparison With the Paper](#5-comparison-with-the-paper)
6. [Why the Gap Remains](#6-our-assumption-about-why-the-gap-remains)
7. [Next Steps](#7-next-steps)
8. [Conclusion](#8-conclusion)

---

## 1. Problem Statement

- Fake faces (deepfakes) are getting more realistic and harder to spot.
- New ways to make fakes appear all the time, each leaving different traces.
- A detector trained on old fakes often fails on new ones.
- Retraining on all old and new data every time is slow and costly.
- Training only on the new fakes makes the model forget the old ones. This is called **catastrophic forgetting**.
- Old data cannot always be stored or reused.

### Our Goal

- Build a detector that learns new fake types one after another.
- It should keep detecting the old ones, **with no replay of old data**.
- We follow the DevFD paper: a growing set of small **LoRA experts** with an **orthogonal constraint**.

---

## 2. Core Idea Behind DevFD

![DevFD architecture](assets/devfd_architecture.png)

### 2.1 The overall network (left of the figure)

| Component | What it does |
| --- | --- |
| **Input Faces** | The face photos to check (bottom of the figure). |
| **Patch Embedding** | Cuts each image into small squares and turns them into numbers. |
| **Layer Norm + Multi-Head Self Attention** | Standard transformer parts. They help the model see how different parts of the face relate to each other. |
| **Feed Forward (FFN)** | The normal part of each block. It is **frozen**, so it does not change. |
| **Developmental MoE** (red box) | The **new part**. It sits next to the FFN, and their outputs are added together. |
| **× N** | The whole block repeats many times. |
| **Classifier** | Gives the final answer: **Real** or **Fake**. The flame icon means it is trainable. |

### 2.2 Developmental MoE (middle of the figure)

MoE means **"mixture of experts"**. It is a group of small helper modules, each called a **LoRA**.

- **Red LoRA (Real-LoRA):** one helper that studies real faces. It keeps learning in every task, so it has a flame 🔥.
- **Blue LoRAs (Fake-LoRAs):** one helper for each new type of fake, named `T1, T2, … Ti` (task 1, task 2, and so on).
- **Snowflake icon ❄️ = frozen.** Old Fake-LoRAs are frozen so they cannot forget what they learned.
- **Flame on the last blue one** means only the newest Fake-LoRA is trainable.
- When a new kind of fake appears, the model **adds one more blue helper**. That is why it is called *developmental*: it grows over time.
- **Label-guided Localized Balancing** (purple box) mixes the outputs of all helpers into one answer.
- **Shared Subspace vs Orthogonal Subspace:** the real helper has one shared space. Each fake helper gets its own separate space.

### 2.3 Label-guided localized balancing loss — `L_llb` (top right of the figure)

A rule that tells each helper what to focus on.

- **Real face:** the Real-LoRA should respond strongly (**High**). The Fake-LoRAs should respond weakly (**Low**).
- **Fake face:** the opposite. The Fake-LoRAs respond strongly and the Real-LoRA responds weakly.
- This keeps the helpers in their own jobs and stops them from overlapping.

### 2.4 Orthogonal loss — `L_ort` (bottom right of the figure)

*Orthogonal* means pointing in different directions, at right angles, like the corner of a room.

The picture shows three things:

1. **Previous LoRA space** – what the model already knows.
2. **Gradient space** – the direction new learning would push in.
3. **Current LoRA space** – the new helper's space.

The rule makes the new helper learn in a direction that does **not** cross the old ones.

**Why it matters:** if new learning does not overlap old learning, the model does not forget the old fakes.

---

## 3. First Implementation

### Phase I & II – Dataset collection and pre-processing

- We first collected and organised the dataset from the sources named in the paper: **FaceForensics++, DFDC-P, DFD, CDF2**. 
- We wrote an **automation script** that extracts **20 frames** from each of **50 randomly selected videos** (real and the different forged types) and crops them to **224 × 224** pixels to remove noise.
- The result is a clean, ready-to-use dataset, split **80 : 20** (80 % training, 20 % testing).
- Every split has an **equal proportion of real and fake images** to avoid bias.

### Phase III – Importing the pre-built model and training

- We imported the pre-trained **CLIP ViT-B/16** backbone, as used in the paper.
- Using the formulas from the paper, we added DevFD's core capability: **remember old weights** (to detect older forgeries) while **learning new weights** (to detect new forgeries).
- We trained on our structured dataset.

![LoRA forward formula](assets/formula_lora_forward.png)

| Symbol | Meaning |
| --- | --- |
| `x` | Input going into a layer |
| `e` | Output coming out |
| `W` | Original **frozen** weight of the layer – it never changes |
| `AⱼBⱼ` | Small LoRA add-on for task *j* (two small matrices multiplied together) |
| `t` | Current task number |

### Phase IV – Reducing loss and the forgetting factor

We used the hyperparameters given in the paper. The objective loss that is minimised:

![Total loss](assets/formula_total_loss.png)

| Term | Meaning |
| --- | --- |
| `L_cls` | **Classification loss** – checks whether the model correctly says real or fake (standard binary cross-entropy). |
| `L_ort` | **Orthogonal loss** – pushes the new Fake-LoRA to learn at right angles to the old ones, so new learning doesn't overwrite old knowledge. |
| `L_llb` | **Label-guided localized balancing loss** – makes the Real-LoRA respond more to real faces and the Fake-LoRAs more to fake faces. |
| `λ₃` | Weight controlling how much `L_llb` counts. The paper uses **0.2**, so it matters less than the other two. |

Next, we wanted to reduce the **forgetting factor** as much as possible, so the model keeps old detection skills while learning new ones.

![Average forgetting](assets/formula_average_forgetting.png)

| Symbol | Meaning |
| --- | --- |
| `AF_T` | Average forgetting after learning *T* tasks in total. **Lower is better.** |
| `T` | Total number of tasks. In our setup **T = 4** (FF++, DFDC-P, DFD, CDF2). |
| `i` | Index of an old task, from 1 to *T − 1*. The last task is left out because it has had no chance to be forgotten yet. |
| `a_{i,i}` | Accuracy on task *i* right after learning task *i* (the diagonal of the accuracy table). |
| `a_{T,i}` | Accuracy on task *i* at the very end, after learning all *T* tasks (the last row of the table). |
| `(a_{i,i} − a_{T,i})` | Accuracy task *i* lost between learning it and the end – the forgetting for that one task. |
| `Σ (i = 1 … T−1)` | Adds up the forgetting of all old tasks. |
| `1 / (T − 1)` | Divides by the number of old tasks, turning the sum into an average. |

### Phase V – Testing (first run)

After training on all four forgery types:

| Task | Accuracy | AUC |
| --- | --- | --- |
| Task 1 | 53.63 % | 89.26 % |
| Task 2 | 62.66 % | 93.92 % |
| Task 3 | 85.75 % | 96.97 % |
| Task 4 | 99.50 % | 99.94 % |

- **Average Accuracy (AA): 75.38 %**
- **Average Forgetting (AF): 30.65 %**

> **Too bad!!** – far from the paper's numbers.

---

## 4. Second Implementation – Finding the Real Problem

### Phase VI – Going back to the paper

We had drifted from the paper's actual implementation, so we re-read it carefully and applied these fixes:

- Fake-LoRA 1 is created in Task 1. The Real-LoRA stays trainable in all tasks.
- Only **old** Fake-LoRAs are frozen when a new task starts.
- Old routing weights are not updated when a new expert is added.
- Every batch has exactly **16 real and 16 fake** images.
- CLIP normalisation is used for input images.
- The λ₁/λ₂ schedule follows the paper (0.5/0.5, then 1/0.1, then 1/0.01).
- The balancing loss is computed **per image**, as in the paper.
- The orthogonal-gradient basis is computed from the full input matrix (SVD, done through its 768 × 768 covariance for speed).

### Phase VII – Choosing the LoRA rank

The paper picks the rank by grid search but does not list the values, so we tested several ranks on a validation slice taken from the training data.

| Rank | AA | AF |
| --- | --- | --- |
| 4 | 74.02 | 32.02 |
| 8 | 77.04 | 28.45 |
| 16 | 75.99 | 30.62 |
| **32** | **77.40** | **28.03** ✅ best |
| 64 | 75.69 | 30.37 |
| 128 | 74.05 | 31.91 |

**Rank 32 is the best.**

### Phase VIII – A bug we found in the orthogonal loss

- The orthogonal loss stayed stuck at about **7.87** and did not go down.
- **Reason:** the gradient part of the loss used only frozen, detached values, so no trainable weight could change it.
- **Fix:** we kept only the subspace part, which does depend on the new LoRA.
- After the fix, the loss started small (**0.0007**) and went to **0** within a few epochs.

**Potential risk it could have caused:** a loss that never changes is a warning sign. Here it meant part of the loss was not teaching the model anything.

### Phase IX – SBI pre-training

The paper first pre-trains the backbone on real faces using the **self-blended images (SBI)** idea, but does not give the settings.

- **SBI:** blend a slightly altered copy of a real face back onto the same face. The seams act as "fake" examples.
- **Data:** real FF++ videos that were **not** among our 50 chosen videos. No fake videos were used.
- Faces cropped with **MTCNN** and a 15 % margin (same as our training data).
- **8000** crops used; face landmarks found for **7973 / 8000 (99.7 %)**.
- **10 epochs**, learning rate **1e-5**. Accuracy rose from **93.6 % to 99.2 %**.

![SBI pre-training samples – real face crops (top row) and their self-blended versions (bottom row)](assets/sbi_pretraining.png)

*Top row: real face crops. Bottom row: the same faces after self-blending.*

### Phase X – Result with SBI backbone (rank 32)

| Task | Accuracy | AUC |
| --- | --- | --- |
| Task 1 | 52.99 % | 86.86 % |
| Task 2 | 66.17 % | 88.58 % |
| Task 3 | 81.50 % | 94.14 % |
| Task 4 | 99.00 % | 99.98 % |

- **AA: 74.91 %** · **AF: 29.59 %** → only a **minute improvement** !

SBI pre-training did **not** reduce forgetting. The pattern is the same: **high AUC on old tasks, low accuracy**.

### Phase XI – Testing the classifier head

**Idea:** maybe the classifier at the top drifts. We **froze it after Task 1**.

| Task | Accuracy | AUC |
| --- | --- | --- |
| Task 1 | 51.82 % | 85.91 % |
| Task 2 | 66.42 % | 88.97 % |
| Task 3 | 82.75 % | 94.53 % |
| Task 4 | 99.50 % | 99.98 % |

- **AA: 75.12 %** · **AF: 29.39 %**

**Result:** almost no change (AF 29.59 % → 29.39 %), so the classifier is **not** the main cause.

---

## 5. Comparison With the Paper

| Run | AA (↑) | AF (↓) |
| --- | --- | --- |
| **Paper (DevFD)** | **89.82** | **4.03** |
| Our first run | 75.38 | 30.65 |
| + SBI pre-training | 74.91 | 29.59 |
| + frozen head | 75.12 | 29.39 |

💔 **We are still too far from the result reported in the paper.**

---

## 6. Our Assumption About Why the Gap Remains

- The paper pre-trains its own backbone with SBI; our version is **simplified**.
- The paper trains on **100 videos × 20 frames per task with batch 128**. We train on more frames with **batch 32**, so the shared parts drift more. We can't use batch 128 because our GPUs crash for batches larger than 32.
- The paper's router is described only briefly. **Ours is our own design.**
- Our split is **by frame**, and the paper's exact test splits are not available.
- We use **one run and one seed**. (The paper gives no error bars either.)

---

## 7. Next Steps

- Freeze the Real-LoRA after Task 1 *(planned test – no result yet)*.
- Test whether adding new experts shifts the old routing weights *(our suspicion, not a proven cause)*.
- Match the paper's data budget: **100 videos × 20 frames, batch 128**.
- Split train and test **by video**, not by frame.

> ⚠️ All of these plans require a **higher grade of hardware** to be implemented.

---

## 8. Conclusion

- We rebuilt DevFD: **CLIP ViT-B/16, LoRA experts, label-guided balancing and orthogonal loss**.
- Our model learns each new task well (**~99 % on the last task**).
- It still forgets old tasks: **AF ≈ 29 % vs 4.03 % in the paper**.
- We ruled out two causes: **rank choice** and **the classifier head**.
- Remaining differences from the paper are being tested.

---

## Reference

- *DevFD: Developmental Face Forgery Detection by Learning Shared and Orthogonal LoRA Subspaces* – <https://arxiv.org/pdf/2509.19230>
- Datasets: FaceForensics++, DFDC-P, DFD, Celeb-DF v2 (CDF2)
