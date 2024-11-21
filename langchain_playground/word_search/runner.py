# This will eventually morph into a Jupyter notebook

from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=.5)

from word_search_builder_v1 import AgentWordSearch
ws = AgentWordSearch()
n_rows = 7
ws.create_word_search(model, 'math', 5, n_rows, n_rows)

print(ws.words)
print(ws.puzzle)

from word_search_checker import string_to_grid, find_words, get_puzzle_accuracy

word_matches = find_words(string_to_grid(ws.puzzle), ws.words)
print(word_matches)

# Get % words with exactly one match
get_puzzle_accuracy(word_matches)

# Simulate 10 runs
get_puzzle_accuracies = []
for i in range(10):
    ws = AgentWordSearch()
    ws.create_word_search(model, 'math', 5, n_rows, n_rows)
    try:
        word_matches = find_words(string_to_grid(ws.puzzle), ws.words)
    except:
        # any puzzle error means accuracy = 0
        get_puzzle_accuracies.append(0)
    else:
        get_puzzle_accuracies.append(get_puzzle_accuracy(word_matches))
    
    # Print status
    if i % 5 == 4:
        print(i+1)

import numpy as np
print(f'''
    # Simulations: {len(get_puzzle_accuracies)}
    Accuracies: {get_puzzle_accuracies}
    Median: {np.median(get_puzzle_accuracies)}
    Mean: {np.mean(get_puzzle_accuracies)}
''')
