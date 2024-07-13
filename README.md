# Markov Text Generation 

This project demonstrates text generation using Markov chains, a probabilistic method where the next token (word or character) in a sequence is predicted based on the current state of the sequence. It includes both word-level and character-level Markov chain models and predicts the probability of the next token given a current state.

## Introduction

Markov chains are powerful tools for generating text based on statistical patterns observed in a given corpus. This project showcases how to implement Markov chain models for text generation and probability prediction.
In this project, we will:
- Implement both word-level and character-level Markov chain models for text generation.
- Calculate probabilities of the next token based on the current state in the Markov chain.
- Generate sequences of text based on a given initial seed.
- Write the generated sequences to an output file (`result.txt`).
- Provide an example of running the script with input text and viewing generated results.

These aspects of the project showcase how Markov chains can be applied to generate coherent sequences of text from a given corpus, whether for creative writing, data augmentation, or other text generation tasks.

## Prerequisites

- Install Python (3.6 or higher is recommended) from their official website.
(https://www.python.org/downloads/)

## Setup

**Create a Project Folder**:
   - Create a new folder for your project. You can name it as per your preference.

## Input Text

Create a file named `input.txt` in the project directory and add your text corpus to it. Ensure the text is formatted as desired for text generation.

## Code
Create a Python script named markov_text.py in your project folder and add the following code in it.

```python
import string
from collections import defaultdict, Counter
import random

# Specify the full path to input.txt
input_file_path = 'your_input_file_path.txt'

# Read text from input file
with open(input_file_path, 'r') as file:
    text = file.read()

# Clean the text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace('\n', ' ')  # Replace newline characters with space
    return text

# Tokenization (default to word-level, can switch to character-level)
def tokenize_text(text, level='word'):
    if level == 'word':
        return text.split()  # Split text into words
    elif level == 'char':
        return list(text)  # Split text into characters

# Build Markov chain model
class MarkovChain:
    def __init__(self, order=3):
        self.order = order
        self.model = defaultdict(Counter)

    def add_sequence(self, tokenized_text):
        for i in range(len(tokenized_text) - self.order):
            current_state = tuple(tokenized_text[i:i + self.order])
            next_state = tokenized_text[i + self.order]
            self.model[current_state][next_state] += 1

    def get_next_token_probability(self, current_state, next_token):
        transitions = self.model.get(current_state, Counter())
        total = sum(transitions.values())
        if total == 0:
            return 0.0
        return transitions.get(next_token, 0) / total

    def generate_sequence(self, seed, length=50):
        current_state = seed
        result = list(current_state)
        for _ in range(length):
            transitions = self.model.get(current_state, None)
            if not transitions:
                break  # Exit loop if current state has no transitions
            next_token = random.choices(list(transitions.keys()), 
                                        list(transitions.values()))[0]
            result.append(next_token)
            current_state = tuple(result[-self.order:])
        return result

# Main function to execute the entire process
def main():
    cleaned_text = clean_text(text)
    tokenized_text_word = tokenize_text(cleaned_text, level='word')
    tokenized_text_char = tokenize_text(cleaned_text, level='char')

    # Word-level Markov chain model
    markov_model_word = MarkovChain(order=3)
    markov_model_word.add_sequence(tokenized_text_word)
    seed_word = tuple(tokenized_text_word[:3])  # Use the first three words as the initial state
    generated_sequence_word = markov_model_word.generate_sequence(seed_word, length=50)

    # Character-level Markov chain model
    markov_model_char = MarkovChain(order=4)
    markov_model_char.add_sequence(tokenized_text_char)
    seed_char = tuple(tokenized_text_char[:4])  # Use the first four characters as the initial state
    generated_sequence_char = markov_model_char.generate_sequence(seed_char, length=200)

    # Prepare output
    output = []
    output.append("Generated Word Sequence:")
    output.append(' '.join(generated_sequence_word))
    output.append("\nGenerated Character Sequence:")
    output.append(''.join(generated_sequence_char))

    # Predict next token probabilities for multiple words and characters
    word_probabilities = [
        (('alice', 'was', 'beginning'), 'to'),
        (('to', 'get', 'very'), 'tired'),
        (('and', 'of', 'having'), 'nothing'),
        (('she', 'had', 'peeped'), 'into'),
        (('into', 'the', 'book'), 'her'),
        (('book', 'her', 'sister'), 'was'),
        (('her', 'sister', 'was'), 'reading'),
        (('but', 'it', 'had'), 'no'),
        (('no', 'pictures', 'or'), 'conversations'),
        (('thought', 'alice', 'without'), 'pictures'),
        # Add more (current_state, next_token) tuples as needed
    ]

    char_probabilities = [
        (('a', 'l', 'i', 'c'), 'e'),
        (('t', 'h', 'e', ' '), 'b'),
        (('o', 'n', 'c', 'e'), ' '),
        (('h', 'e', 'r', ' '), 's'),
        (('t', 'h', 'o', 'u'), 'g'),
        (('r', 'e', 'a', 'd'), 'i'),
        (('b', 'o', 'o', 'k'), ' '),
        (('c', 'o', 'n', 'v'), 'e'),
        (('s', 'i', 's', 't'), 'e'),
        (('a', 'n', 'd', ' '), 'o'),
        # Add more (current_state, next_token) tuples as needed
    ]

    for current_state_word, next_token_word in word_probabilities:
        next_token_prob_word = markov_model_word.get_next_token_probability(current_state_word, next_token_word)
        output.append(f"\nProbability of '{next_token_word}' following {current_state_word}: {next_token_prob_word}")

    for current_state_char, next_token_char in char_probabilities:
        next_token_prob_char = markov_model_char.get_next_token_probability(current_state_char, next_token_char)
        output.append(f"\nProbability of '{next_token_char}' following {current_state_char}: {next_token_prob_char}")

    # Write results to output file
    with open('result.txt', 'w') as file:
        file.write('\n'.join(output))

if __name__ == "__main__":
    main()
```

### Running the Script

To run the script that generates text using Markov chains:

```sh
python markov_text.py
```

The script processes the text from `input.txt`, builds Markov chain models for both word-level and character-level sequences, generates output text, and calculates next token probabilities.

### Example

For example, if `input.txt` contains:

```
Alice was beginning to get very tired of sitting by her sister on the bank, 
and of having nothing to do: once or twice she had peeped into the book her 
sister was reading, but it had no pictures or conversations in it, 'and what 
is the use of a book,' thought Alice 'without pictures or conversation?'
```

The output generated in `result.txt` might be:

```
Generated Word Sequence:
alice was beginning to get very tired of sitting by her sister on the bank and of having nothing to do once or twice she had peeped into the book her sister was reading but it had no pictures or conversations in it and what is the use of a book thought alice

Generated Character Sequence:
alice with pink eyes ran close by her feel very sleepy and stupid when suddenly a white rabbit withought alice was reading nothing to do once or conversation so she had no pictures or the bank and picture

Probability of 'to' following ('alice', 'was', 'beginning'): 1.0

Probability of 'tired' following ('to', 'get', 'very'): 1.0

Probability of 'nothing' following ('and', 'of', 'having'): 1.0

Probability of 'into' following ('she', 'had', 'peeped'): 1.0

Probability of 'her' following ('into', 'the', 'book'): 1.0

Probability of 'was' following ('book', 'her', 'sister'): 1.0

Probability of 'reading' following ('her', 'sister', 'was'): 1.0

Probability of 'no' following ('but', 'it', 'had'): 1.0

Probability of 'conversations' following ('no', 'pictures', 'or'): 1.0

Probability of 'pictures' following ('thought', 'alice', 'without'): 1.0

Probability of 'e' following ('a', 'l', 'i', 'c'): 1.0

Probability of 'b' following ('t', 'h', 'e', ' '): 0.2857142857142857

Probability of ' ' following ('o', 'n', 'c', 'e'): 1.0

Probability of 's' following ('h', 'e', 'r', ' '): 0.4

Probability of 'g' following ('t', 'h', 'o', 'u'): 0.5

Probability of 'i' following ('r', 'e', 'a', 'd'): 1.0

Probability of ' ' following ('b', 'o', 'o', 'k'): 1.0

Probability of 'e' following ('c', 'o', 'n', 'v'): 1.0

Probability of 'e' following ('s', 'i', 's', 't'): 1.0

Probability of 'o' following ('a', 'n', 'd', ' '): 0.25
```

### Contributing

Contributions to improve this project are welcome! If you'd like to contribute:

- Fork the repository.
- Create a new branch with a descriptive name (`git checkout -b my-branch-name`).
- Make your changes and commit them (`git commit -am 'Add some feature'`).
- Push to the branch (`git push origin my-branch-name`).
- Create a new Pull Request.

Please adhere to the project's coding standards and include relevant tests with your contributions.
