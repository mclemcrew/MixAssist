# MixAssist: An Audio-Language Dataset for Co-Creative AI Assistance in Music Mixing

By: [Michael Clemens](http://mclem.in), [Ana Marasović](https://www.anamarasovic.com)

![A diagram illustrating an interaction between an expert and an amateur while discussing an audio segment. At the center is an audio waveform labeled "Audio segment being discussed." On the left is a person labeled "Expert." An arrow shows they "Listen alongside amateur." The expert says, "I think that tom could be EQ'd a little bit, just a low shelf around 150Hz and bring it down maybe like 6dBs," and "For this style of music, I would typically probably keep it a little bit more tame."
On the right is a person with headphones labeled "Amateur." An arrow shows they "Play audio relevant to question." The amateur says, "I am just trying to blend in these drums with this a little bit," "Oh! I feel like that's sounding better already," and "This is so fun! I feel like I'm learning so much!"](imgs/dataset-description.png "Dataset Setup")

This repository contains the official evaluation scripts for the paper: **"MixAssist: An Audio-Language Dataset for Co-Creative AI Assistance in Music Mixing"**.

We introduce the **MixAssist** dataset, a novel audio-language resource designed to capture the nuanced, multi-turn dialogue between expert and amateur producers during live mixing sessions.

## About The Project

While many AI tools automate mixing, they often provide limited pedagogical value for artists who want to improve their skills. MixAssist aims to foster the development of intelligent AI assistants that can support and augment the creative process through contextual, instructional dialogue.

This repository provides the code to replicate the **"LLM-as-a-Judge"** evaluation methodology we used to assess and rank the performance of different audio-language models fine-tuned on the MixAssist dataset.

## The Datasets

Our work introduces two new resources to facilitate research in this area:

### 1. The MixAssist Dataset

![A diagram illustrating a data processing pipeline in five stages, moving from left to right. **Stage 1: Raw Data.** Icons of two-person groups (an amateur and an expert) are shown next to a "REC" symbol, indicating 90-minute recording sessions. **Stage 2: Conversation Transcripts.** An audio waveform is displayed above corresponding transcribed text, such as "And overheads..." and "...It's pop rock.” **Stage 3: Speaker-Split & Audio-Aligned.** The conversation is broken down into turns, with each line of text assigned to a speaker (A for amateur, E for expert) and aligned with its specific audio segment. **Stage 4: Focused Sub-Conversations.** The conversations are grouped into thematically focused sub-conversations, represented by icons like a piano or drums. Each group contains a sequence of speaker-and-audio-aligned turns. **Stage 5: Dataset Instances.** This shows the final structure for a machine learning model. The model input consists of audio and text (including a prior conversation summary and utterances from the amateur and expert). The model output is the subsequent expert utterance.](imgs/data-flow.png "Data Flow")

The primary dataset, MixAssist, captures the conversational "why" behind mixing decisions.

- **Content**: The dataset consists of 431 audio-grounded, multi-turn conversational turns derived from 7 in-depth mixing sessions involving 12 unique producers.
- **Structure**: It features natural, instructional dialogue between expert and amateur producers, temporally aligned with the specific audio segments being discussed. The dataset is filtered to focus on pedagogically valuable interactions.
- **Availability**:
  - The processed conversational dataset is available on [Hugging Face](https://huggingface.co/datasets/mclemcrew/MixAssist).
  - The raw, unprocessed session recordings (dialogue and DAW playback), aligned music-only audio segments, and transcripts of these sessions are available on [Zenodo]().

### 2. The MixParams Dataset

As a complementary resource, we also introduce MixParams, which focuses on the technical "how" of mixing.

- **Content**: A detailed dataset containing specific parameter settings (e.g., EQ, compression) for 114 individual mixes derived from the same source songs used in the MixAssist sessions.
- **Purpose**: This dataset enables research into tasks like parameter prediction and provides a bridge between high-level instructional dialogue and concrete technical implementation in a DAW.
- **Availability**: The MixParams dataset is available on [Hugging Face](https://huggingface.co/datasets/mclemcrew/MixParams).

## Evaluation using LLM-as-a-Judge

The scripts in this repository (`llm_judge.py` and `run_judge.py`) implement the evaluation framework used in our paper to compare the fine-tuned models (Qwen-Audio, LTU, and MU-LLaMA).

We used an "LLM-as-a-Judge" approach with `o3-mini` to rank the three model outputs for each test prompt. This listwise ranking forces a clear preference ordering based on predefined criteria. The judge evaluates responses based on three critical criteria in order of importance: technical accuracy, helpfulness, and conversation fluency.

- **`llm_judge.py`**: A class that encapsulates the logic for the judge, including the detailed system prompt, API handling for different models (OpenAI, local models via Ollama), and result processing.
- **`run_judge.py`**: A command-line interface to execute the evaluation on a dataset, run tests, and generate result visualizations and reports.

#### Prerequisites

- An API key and endpoint for your chosen judge model (e.g., OpenAI API key for `gpt-4o` or a local Ollama endpoint for `qwen`).

### Usage

You can run a small test or a full evaluation using the `run_judge.py` script.

**To run a quick test on 5 random samples from your data:**

```sh
python run_judge.py --data "path/to/your/dataset.csv" --samples 5 --model "gpt-4o"
```

_(After the initial test run, the script will prompt you to proceed with the full dataset evaluation.)_

<!-- #### How to Cite
If you use the MixAssist or MixParams dataset, please cite our paper:

@inproceedings{anonymous2025mixassist,
    title={{MixAssist: An Audio-Language Dataset for Co-Creative AI Assistance in Music Mixing}},
    author={Anonymous},
    booktitle={Under review as a conference paper at COLM 2025},
    year={2025},
    url={https://your_paper_link_here}
} -->
