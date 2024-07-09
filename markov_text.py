import string
from collections import defaultdict, Counter
import random

# Read text from input file
with open('input.txt', 'r') as file:
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

    # Predict next token probabilities
    current_state_word = ('alice', 'was', 'beginning')  # Example current state for word-level
    next_token_prob_word = markov_model_word.get_next_token_probability(current_state_word, 'to')
    output.append(f"\nProbability of 'to' following {current_state_word}: {next_token_prob_word}")

    current_state_char = ('a', 'l', 'i', 'c')  # Example current state for character-level
    next_token_prob_char = markov_model_char.get_next_token_probability(current_state_char, 'e')
    output.append(f"\nProbability of 'e' following {current_state_char}: {next_token_prob_char}")

    # Write results to output file
    with open('result.txt', 'w') as file:
        file.write('\n'.join(output))

if __name__ == "__main__":
    main()
