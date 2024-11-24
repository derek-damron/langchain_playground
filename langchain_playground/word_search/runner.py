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

from word_search_checker import string_to_grid, find_words, get_puzzle_accuracy, check_puzzle_size

# Check size
check_puzzle_size(string_to_grid(ws.puzzle), n_rows, n_rows)

# Get % words with exactly one match
word_matches = find_words(string_to_grid(ws.puzzle), ws.words)
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

# Test a different prompt
from langchain_core.prompts import PromptTemplate
alt_puzzlemaker_prompt = PromptTemplate(
    template = """
    You are an expert word search puzzle creator. Your task is to generate a word search puzzle based on the given input. Here are the details:

    Instructions:
    1. Create a word search puzzle using these words: {words}
    2. The puzzle grid should have {n_rows} rows and {n_cols} columns.
    3. Place each word in the puzzle exactly once.
    4. Words must be placed in straight lines: horizontally, vertically, or diagonally.
    5. Words must not pivot or change direction within the puzzle.
    6. Do not insert extra letters into words.
    7. Fill empty spaces with random letters.

    Puzzle Creation Process:
    1. Create an empty grid of the specified size.
    2. For each word:
        a. Choose a random starting position.
        b. Choose a random direction (horizontal, vertical, or diagonal).
        c. Check if the word fits in the chosen direction without overlapping other words.
        d. If it fits, place the word; if not, try a new position or direction.
    3. Once all words are placed, fill remaining spaces with random letters.

    EXAMPLE 1:
    Words are dog, cat, ant
    Puzzle is 5 rows and 5 columns
    Example puzzle is
    X X X X D
    X T X X O
    T N X X G
    X A X X X
    X X C X X

    Output Format: Return only the puzzle with one space between each letter.  Do not provide any other output such as "Here's a 7X7 word search puzzle grind containing the requested words:"
    """,
    input_variables = ['words', 'n_rows', 'n_cols']
)
puzzle_accuracies = []
model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
for i in range(10):
    ws = AgentWordSearch()
    ws.create_word_search(model, 'math', 5, n_rows, n_rows, puzzlemaker_prompt=alt_puzzlemaker_prompt)
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

print(f'''
    # Simulations: {len(puzzle_accuracies)}
    Accuracies: {puzzle_accuracies}
    Median: {np.median(puzzle_accuracies)}
    Mean: {np.mean(puzzle_accuracies)}
''')
