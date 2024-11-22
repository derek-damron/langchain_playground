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
puzzle_accuracies = []
for i in range(10):
    ws = AgentWordSearch()
    ws.create_word_search(model, 'math', 5, n_rows, n_rows)
    try:
        word_matches = find_words(string_to_grid(ws.puzzle), ws.words)
    except:
        # any puzzle error means accuracy = 0
        puzzle_accuracies.append(0)
    else:
        puzzle_accuracies.append(get_puzzle_accuracy(word_matches))
    
    # Print status
    if i % 5 == 4:
        print(i+1)

import numpy as np
print(f'''
    # Simulations: {len(puzzle_accuracies)}
    Accuracies: {puzzle_accuracies}
    Median: {np.median(puzzle_accuracies)}
    Mean: {np.mean(puzzle_accuracies)}
''')


# Simulate 10 runs at three different temperatures
puzzle_accuracies = []
for tmp in [0, 0.25, .5]:
    print(f'''Temperature: {tmp}''')
    model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=tmp)
    puzzle_accuracies = []
    for i in range(10):
        ws = AgentWordSearch()
        ws.create_word_search(model, 'math', 5, n_rows, n_rows)
        try:
            word_matches = find_words(string_to_grid(ws.puzzle), ws.words)
        except:
            # any puzzle error means accuracy = 0
            puzzle_accuracies.append(0)
        else:
            puzzle_accuracies.append(get_puzzle_accuracy(word_matches))
    print(f'''
        # Simulations: {len(puzzle_accuracies)}
        Accuracies: {puzzle_accuracies}
        Median: {np.median(puzzle_accuracies)}
        Mean: {np.mean(puzzle_accuracies)}
    ''')
    puzzle_accuracies = []

