<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=✍️%20Handwritten%20Word%20%26%20Sentence%20Reader&fontSize=34&fontColor=fff&animation=twinkling&fontAlignY=35&desc=CRNN%20%2B%20CTC%20Deep%20Learning%20%7C%20CodeAlpha%20ML%20Internship&descAlignY=55&descSize=17" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Gradio](https://img.shields.io/badge/Gradio-Web%20App-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)

[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![Architecture](https://img.shields.io/badge/Architecture-CRNN-9B59B6?style=for-the-badge)](.)
[![License](https://img.shields.io/badge/License-Educational-27AE60?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete%20✅-2ECC71?style=for-the-badge)](.)

<br/>

> **"This system does not just guess one letter — it reads a whole word, the same way a human eye scans a sentence."**

<br/>

**Developer:** Taj Wali Khan &nbsp;|&nbsp;
**Internship:** CodeAlpha Machine Learning — Task 3 &nbsp;|&nbsp;
**Platform:** Google Colab

</div>

---

## 📌 Table of Contents

- [What Is This Project?](#-what-is-this-project)
- [Why This Is Different From a Normal Letter Classifier](#-why-this-is-different-from-a-normal-letter-classifier)
- [How the Training Data Is Made](#-how-the-training-data-is-made)
- [The Big Idea: CTC Loss, Explained Simply](#-the-big-idea-ctc-loss-explained-simply)
- [Model Architecture (CRNN)](#-model-architecture-crnn)
- [Data Augmentation — Making Fake Handwriting Look Real](#-data-augmentation--making-fake-handwriting-look-real)
- [Training Strategy](#-training-strategy)
- [How We Measure Success: CER and WER](#-how-we-measure-success-cer-and-wer)
- [Results](#-results)
- [The Web App](#-the-web-app)
- [Quick Start — Run It Yourself](#-quick-start--run-it-yourself)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Full Pipeline at a Glance](#-full-pipeline-at-a-glance)
- [Limitations & Honest Notes](#-limitations--honest-notes)
- [Acknowledgements](#-acknowledgements)

---

## 🌟 What Is This Project?

Most "handwriting recognition" projects you see online only do one small thing: look at a single letter or digit and guess what it is. That is a useful starting point, but it is not how reading actually works in real life.

This project goes a step further. You hand it an image of a handwritten **word** or a **short phrase** — something like `"machine learning"` — and it reads the whole thing back to you as text, letter by letter, in the correct order.

```
Input:   [ image of someone's handwriting: "happy" ]
Output:  "happy"
```

To do this, the project uses a well-known architecture from real-world Optical Character Recognition (OCR) systems called a **CRNN** (Convolutional Recurrent Neural Network), trained with something called **CTC Loss**. Both ideas are explained in plain English further down this page — no PhD required.

---

## 🆚 Why This Is Different From a Normal Letter Classifier

<table>
<tr><th>A Normal Letter Classifier</th><th>This Project (CRNN + CTC)</th></tr>
<tr>
<td>

- Looks at **one** image
- Picks **one** answer from a fixed list (like `0–9` or `a–z`)
- Easy, but only solves a small piece of the real problem
- Cannot read a whole sentence

</td>
<td>

- Looks at **one** image
- Outputs a **whole string of characters**, like `"good morning"`
- Does not know in advance where one letter ends and the next starts
- Can be extended to full sentences

</td>
</tr>
</table>

This is the exact direction the official task description points toward:

> *"Extendable to full word or sentence recognition with sequence modeling (like CRNN)."*

So instead of building the simple version and stopping there, this project jumps straight to the advanced, more useful version.

---

## 🖋️ How the Training Data Is Made

Getting thousands of real photographs of human handwriting, each correctly labeled, takes a lot of time and money. So instead, this project makes its own training data — a technique real OCR companies also use to bootstrap their first models.

```
Step 1:  Pick a random word or short phrase
         ("happy", "good morning", "open the door")

Step 2:  "Type" it onto a blank image using a real
         handwriting-style font (downloaded for free
         from Google Fonts)

Step 3:  Make it messy and realistic:
         → add gentle wobble to the strokes (elastic warp)
         → rotate it slightly
         → add a touch of noise and blur
         → randomize the ink darkness

Step 4:  Resize it to a standard size the model expects
```

Every single image generated this way looks slightly different — exactly like how no two samples of real human handwriting are ever perfectly identical.

```
        "happy"  (attempt 1)      "happy"  (attempt 2)      "happy"  (attempt 3)
        ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
        │   slight tilt    │      │   different font  │      │   small wobble   │
        │   light ink      │      │   bolder ink       │      │   tiny noise     │
        └─────────────────┘       └─────────────────┘       └─────────────────┘
```

In total, the project generates **15,000 images** (12,000 for training, 1,500 for validation, 1,500 for testing) built from a list of about 100 common English words, mixed into single words and short 2–3 word phrases.

---

## 🧩 The Big Idea: CTC Loss, Explained Simply

This is the single most important concept in the whole project, so let's slow down and walk through it carefully.

### The Problem

Imagine we slice the image of the word **"book"** into 32 thin vertical strips, left to right. We ask the model to guess one letter for every single strip. But "book" only has 4 letters — so many strips end up guessing the same letter over and over:

```
Strip:    1  2  3  4  5  6  7  8  9 10 11 12 ... 32
Guess:    b  b  b  _  _  o  o  o  _  o  o  _ ...  k
```

If we just squash all the repeated letters together naively, we get a problem: `"book"` has **two** o's right next to each other in the correct spelling, but if we blindly merge repeated letters, we would accidentally turn `"book"` into `"bok"`. That's wrong!

### The Solution

CTC solves this with one clever trick: it adds an extra **blank symbol** (think of it like a tiny pause) to the alphabet. The rule becomes:

```
1. Merge letters that repeat *with nothing in between them*
2. BUT if there's a blank symbol between two same letters,
   keep both of them — that blank means "a new letter starts here"
```

So the real, raw output might actually look like this:

```
Strip:    1  2  3  4  5  6  7  8  9 10 11 12 ... 32
Guess:    b  b  b  _  o  o  _  o  o  o  _  k  ...  _
                       ↑              ↑
                  first "o"      blank tells us
                                 second "o" is new
```

After cleaning this up using the blank-aware merging rule: **"book"** ✅

### Why This Matters

The huge benefit is that we **never had to tell the model** exactly which pixel column each letter starts and ends at. That kind of detailed labeling would take forever to create by hand for thousands of images. CTC lets the model figure out this alignment completely on its own, just by seeing the final correct word during training. This is exactly why CTC is the standard choice in both **handwriting recognition** and **speech recognition** systems used in the real world.

---

## 🧠 Model Architecture (CRNN)

CRNN stands for **Convolutional Recurrent Neural Network**. It has two halves that work as a team, each answering a different question:

```
┌──────────────────────────────────────────────────────────────┐
│                 INPUT IMAGE  (32 × 128 pixels)                │
│           a strip of handwriting, like a torn piece of paper  │
└───────────────────────────┬────────────────────────────────────┘
                            │
        ╔═══════════════════╪═══════════════════╗
        ║   PART A — CNN (the "eyes")            ║
        ║   Question: "What do the strokes       ║
        ║              actually look like?"      ║
        ║                                         ║
        ║   Conv2D(32)  → BatchNorm → Pool        ║
        ║   Conv2D(64)  → BatchNorm → Pool        ║
        ║   Conv2D(128) → BatchNorm → Pool (height only) ║
        ║   Conv2D(128) → BatchNorm → Pool (height only) ║
        ║                                         ║
        ║   Output: a grid of detected shapes     ║
        ║   (loops, curves, lines, edges)         ║
        ╚═══════════════════╪═══════════════════╝
                            │
              Turn the grid into a left-to-right
              SEQUENCE of 32 time steps
                            │
        ╔═══════════════════╪═══════════════════╗
        ║   PART B — RNN (the "reading order")   ║
        ║   Question: "What order do these       ║
        ║              shapes come in?"          ║
        ║                                         ║
        ║   Bidirectional LSTM (128 units)        ║
        ║   Bidirectional LSTM (64 units)         ║
        ║   → reads left-to-right AND             ║
        ║     right-to-left at the same time      ║
        ╚═══════════════════╪═══════════════════╝
                            │
              Dense + Softmax layer:
              "At this exact moment, which letter
               (or blank) is most likely?"
                            │
              CTC Decoding → clean final text
                            │
                  "happy"  ✅
```

**Why split the job into two parts like this?** Because recognizing shapes (what the CNN does) and understanding sequence/order (what the RNN does) are genuinely two different skills. Trying to do both with a single type of layer works much worse than letting each part specialize in what it's good at — this is the same idea used in real-world OCR products.

---

## 🎨 Data Augmentation — Making Fake Handwriting Look Real

| Technique | What It Does | Why It Helps |
|-----------|--------------|---------------|
| **Elastic Warp** | Adds a gentle, randomized wobble to the strokes | Real handwriting is never perfectly straight |
| **Random Rotation** (±4°) | Tilts the whole word slightly | People don't always write in a perfectly flat line |
| **Gaussian Noise** | Sprinkles tiny random specks onto the image | Simulates scanner/camera imperfections |
| **Gaussian Blur** | Softens the image slightly | Simulates an out-of-focus photo |
| **Random Ink Darkness** | Changes how dark the "pen" is | Different pens and pressure look different |
| **Random Font + Size** | Picks from 5 different handwriting-style fonts | Different people's handwriting looks different |

---

## 📈 Training Strategy

<div align="center">

| Setting | Value | Why |
|---------|-------|-----|
| Optimizer | Adam, learning rate `1e-3` | Reliable default for most deep learning tasks |
| Loss | **CTC Loss** (built into the model itself) | Lets the model learn without letter-by-letter labels |
| Early Stopping | patience = 10, watches `val_loss` | Stops training once it stops improving |
| Reduce LR on Plateau | factor = 0.5, patience = 4 | Slows down learning when progress stalls, to fine-tune better |
| Batch Size | 64 | Good balance between speed and stable learning |
| Max Epochs | 60 (early stopping usually finishes sooner) | Gives the model plenty of room to improve |
| Live Preview Callback | every 5 epochs | Prints real example reads so you can watch it improve |

</div>

### What You'll See While Training

```
   👀 Sample reads after epoch 5:
         true: 'happy'           →   model said: 'hapy'
         true: 'good morning'    →   model said: 'god mornin'
         true: 'open the door'   →   model said: 'opn the dor'

   👀 Sample reads after epoch 30:
      ✅ true: 'happy'           →   model said: 'happy'
      ✅ true: 'good morning'    →   model said: 'good morning'
         true: 'open the door'   →   model said: 'open the dor'
```

Watching the mistakes shrink epoch by epoch is one of the most satisfying parts of this project.

---

## 📏 How We Measure Success: CER and WER

A simple "accuracy" score doesn't really make sense for this task. Getting one letter wrong out of a 12-letter phrase is a very different (much smaller) mistake than getting the entire word wrong. So instead, the project uses two scorecards that real-world OCR companies use:

```
┌─────────────────────────────────────────────────────────────┐
│  CER — Character Error Rate                                 │
│  "Out of every 100 LETTERS, how many did we get wrong?"     │
│                                                               │
│  True:      h a p p y                                       │
│  Predicted: h a p y                                         │
│  → 1 letter missing out of 5  →  CER = 20%                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  WER — Word Error Rate                                       │
│  "Out of every 100 WHOLE WORDS, how many did we get wrong?"  │
│                                                               │
│  True:      "good morning"                                  │
│  Predicted: "good mornin"                                    │
│  → the second word is wrong → 1 mistake out of 2 words       │
│  → WER = 50%                                                  │
└─────────────────────────────────────────────────────────────┘
```

Both scores are calculated using **Edit Distance** (also called Levenshtein Distance) — counting the minimum number of single-character (or single-word) changes needed to turn the model's guess into the correct answer.

---

## 📊 Results

<div align="center">

| Metric | Meaning | What a Good Score Looks Like |
|--------|---------|------------------------------|
| **Character Error Rate (CER)** | % of individual letters wrong | Lower is better — single-digit % is strong |
| **Word Error Rate (WER)** | % of whole words wrong | Lower is better |
| **Exact Match Rate** | % of predictions that are 100% perfectly correct | Higher is better |

</div>

> Exact figures depend on your specific training run (random seeds, number of epochs reached before early stopping, etc.) — the script prints your actual numbers at the end of Cell 15, and saves a chart of the Character Error Rate distribution across the whole test set.

### What Gets Saved For You to Review

```
/tmp/
├── task3_samples.png             ← examples of generated handwriting
├── task3_training_results.png    ← loss curve + CER distribution chart
├── task3_sample_predictions.png  ← real predictions, green=correct/red=wrong
├── task3_prediction_model.keras  ← the trained model, ready to reuse
└── task3_config.pkl              ← alphabet + settings needed for predictions
```

---

## 🖥️ The Web App

The project ends with a working **Gradio** app you can actually use, with five tabs:

<div align="center">

| Tab | What You Can Do |
|-----|-----------------|
| ✏️ **Read My Handwriting** | Draw a word with your mouse/finger → instant text result + confidence score |
| 📤 **Upload Image** | Upload a photo of handwriting instead of drawing |
| 📊 **Model Performance** | See training curves and the CER score distribution |
| 🖼️ **Sample Training Data** | See what the generated handwriting examples look like |
| ⚙️ **How It Works** | Full plain-English explanation of the architecture, right inside the app |

</div>

### Example Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✍️  HANDWRITING READ RESULT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  📝  Model read:   "good morning"
  📊  Confidence:   87.3%

  🤖  Architecture: CRNN (CNN + BiLSTM + CTC)
  🎯  Test CER:     6.4%
  🎯  Test WER:     14.2%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🚀 Quick Start — Run It Yourself

### Option 1 — Google Colab (Recommended, No Setup Needed)

```
1. Open the script in Google Colab
2. Runtime → Change Runtime Type → T4 GPU   (makes training much faster)
3. Runtime → Run All
4. Wait roughly 20–30 minutes total:
   → a few minutes generating training images
   → 15–25 minutes training the model
5. A public Gradio link appears at the very end — click it to try the app
```

### Option 2 — Run Locally on Your Own Computer

```bash
# 1. Clone the project
git clone https://github.com/YOUR_USERNAME/CodeAlpha_HandwritingWordRecognition.git
cd CodeAlpha_HandwritingWordRecognition

# 2. Set up a clean Python environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install everything the project needs
pip install -r requirements.txt

# 4. Run it
python task_3_handwritten_word_sentence_recognition_crnn.py

# 5. Open the app
# → http://localhost:7860
```

### `requirements.txt`

```txt
tensorflow>=2.12.0
opencv-python-headless>=4.8.0
Pillow>=10.0.0
gradio>=4.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 🗂️ Project Structure

```
CodeAlpha_HandwritingWordRecognition/
│
├── 📄 task_3_handwritten_word_sentence_recognition_crnn.py   ← main script (19 cells)
├── 📄 README.md                                              ← this file
├── 📄 requirements.txt                                       ← dependencies
│
└── 📁 /tmp/  (created automatically when you run the script)
    ├── handwriting_fonts/             ← downloaded handwriting-style fonts
    ├── task3_samples.png              ← sample generated handwriting
    ├── task3_crnn_best.keras          ← best model checkpoint during training
    ├── task3_prediction_model.keras   ← final, ready-to-use model
    ├── task3_config.pkl               ← alphabet + settings for predictions
    ├── task3_training_results.png     ← loss curves + CER chart
    └── task3_sample_predictions.png   ← visual grid of real predictions
```

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | Core development |
| **Deep Learning** | ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white) | Building and training the CRNN |
| **Image Processing** | ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?logo=opencv&logoColor=white) ![Pillow](https://img.shields.io/badge/-Pillow-3776AB?logoColor=white) | Generating and distorting handwriting images |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?logoColor=white) ![Seaborn](https://img.shields.io/badge/-Seaborn-3498DB?logoColor=white) | Charts and result grids |
| **Web App** | ![Gradio](https://img.shields.io/badge/-Gradio-FF7C00?logoColor=white) | The interactive demo |
| **Platform** | ![Colab](https://img.shields.io/badge/-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white) | Free GPU training environment |

</div>

---

## 🧭 Full Pipeline at a Glance

```
╔═══════════════════════════════════════════════════════════════════╗
║                    THE WHOLE PROJECT, START TO FINISH              ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                     ║
║  1. PICK WORDS         → choose from ~100 common English words     ║
║                           mix single words and 2–3 word phrases    ║
║                                  ↓                                  ║
║  2. CREATE IMAGES       → "type" them with handwriting fonts,      ║
║                            then warp/rotate/blur to look real      ║
║                                  ↓                                  ║
║  3. ENCODE LABELS       → turn "happy" into numbers the model      ║
║                            can understand: [7,0,15,15,24]          ║
║                                  ↓                                  ║
║  4. BUILD THE CRNN       → CNN (sees shapes) + BiLSTM (reads order) ║
║                                  ↓                                  ║
║  5. TRAIN WITH CTC LOSS  → model learns its own letter-to-pixel    ║
║                             alignment, no manual labeling needed    ║
║                                  ↓                                  ║
║  6. DECODE PREDICTIONS   → turn raw model output back into text    ║
║                             using CTC's blank-aware merging rule    ║
║                                  ↓                                  ║
║  7. SCORE THE MODEL      → Character Error Rate + Word Error Rate  ║
║                                  ↓                                  ║
║  8. DEPLOY               → Gradio app: draw or upload → read text  ║
║                                                                     ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## ⚠️ Limitations & Honest Notes

- **Lowercase letters only.** The alphabet is `a–z` plus a space character. Capital letters, numbers, and punctuation are not included — though the code is written so you can add them easily by editing one list in the settings section.
- **Trained on synthetic (computer-generated) handwriting**, not real human handwriting photos. It reads clean fonts well; genuinely messy human handwriting may be harder for it, since it never saw real handwriting samples during training.
- **Maximum length of 16 characters** per prediction. Longer sentences would need a bigger model and more training time.
- This is an **educational project** built for the CodeAlpha Machine Learning Internship — not a production-grade OCR product.

---

## 🙌 Acknowledgements

- **[CodeAlpha](https://www.codealpha.tech)** — for designing a task that pushes beyond basic tutorials into real engineering territory
- **[Google Fonts](https://fonts.google.com)** — for the free handwriting-style fonts used to generate training data
- **CTC Loss** — originally introduced by Graves et al. (2006), the foundational technique behind modern speech and handwriting recognition
- **CRNN Architecture** — based on the ideas from Shi et al., *"An End-to-End Trainable Neural Network for Image-based Sequence Recognition"* (2015)
- **[TensorFlow / Keras](https://tensorflow.org)**, **[OpenCV](https://opencv.org)**, **[Gradio](https://gradio.app)** — the open-source tools that made this project possible

---

## 📬 Contact

<div align="center">

| Platform | Link |
|----------|------|
| 🌐 **CodeAlpha** | [www.codealpha.tech](https://www.codealpha.tech) |
| 📧 **Email** | services@codealpha.tech |
| 💬 **WhatsApp** | +91 9336576683 |

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=twinkling" width="100%"/>

**Made with ❤️ by Taj Wali Khan · CodeAlpha ML Internship — Task 3**

*"Teaching a machine to read is just teaching it to listen to a sequence — one letter at a time."*

⭐ **Star this repo** if it helped you understand CRNN and CTC a little better!

</div>
