from typing import List, Tuple, Dict

def find_word(grid: List[List[str]], word: str) -> List[List[Tuple[int, int]]]:
    """
    Find all occurrences of a word in a word search grid.
    Searches in all 8 directions: horizontal, vertical, and diagonal.
    
    Args:
        grid (List[List[str]]): The word search grid as a 2D list of characters
        word (str): The word to search for
        
    Returns:
        List[List[Tuple[int, int]]]: List of found matches, where each match is a list of 
                                    coordinate tuples (row, col) representing the path of the word
    """
    if not grid or not grid[0] or not word:
        return []
    
    rows, cols = len(grid), len(grid[0])
    matches = []
    
    # All possible directions: right, right-down, down, left-down, left, left-up, up, right-up
    directions = [
        (0, 1),   # right
        (1, 1),   # right-down
        (1, 0),   # down
        (1, -1),  # left-down
        (0, -1),  # left
        (-1, -1), # left-up
        (-1, 0),  # up
        (-1, 1)   # right-up
    ]
    
    def is_valid_position(row: int, col: int) -> bool:
        """Check if the given position is within the grid boundaries."""
        return 0 <= row < rows and 0 <= col < cols
    
    def search_direction(start_row: int, start_col: int, direction: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Search for the word starting from a position in a given direction.
        Returns the path of coordinates if found, empty list otherwise.
        """
        dr, dc = direction
        path = []
        row, col = start_row, start_col
        
        for char in word:
            if not is_valid_position(row, col) or grid[row][col] != char:
                return []
            path.append((row, col))
            row += dr
            col += dc
            
        return path
    
    # Search through each position in the grid
    for row in range(rows):
        for col in range(cols):
            # Try each direction from the current position
            for direction in directions:
                path = search_direction(row, col, direction)
                if path:
                    matches.append(path)
    
    return matches

def find_words(grid: List[List[str]], words: List[str]) -> Dict[str, List[List[Tuple[int, int]]]]:
    """
    Wrapper for find_words() on a list of words
    """
    word_matches = dict()
    for w in words:
        word_matches[w] = find_word(grid, w)
    return word_matches

def get_puzzle_accuracy(found_words: Dict[str, List[List[Tuple[int, int]]]]) -> float:
    """
    Args:
        found_words: Dict[str, List[List[Tuple[int, int]]]]: Output of find_words()

    Returns:
        float: % of words with only one match in the puzzle
    """
    only_one_match = [1 if len(v) == 1 else 0 for v in found_words.values()]
    return sum(only_one_match) / len(only_one_match)

def check_puzzle_size(grid: List[List[str]], n_rows: int, n_cols: int) -> bool:
    """
    Args:
        grid: List[List[str]]: Puzzle grid
        n_rows: int: The expected number of rows in the puzzle
        n_cols: int: The expected number of columns in the puzzle

    Returns:
        bool: True is puzzle size is correct, otherwise False
    """
    # Check rows
    if len(grid) != n_rows:
        return False
    
    # Check columns
    if not all([len(l) == n_cols for l in grid]):
        return False

    return True

def print_grid(grid: List[List[str]]) -> None:
    """
    Print the word search grid with spaces between letters.
    
    Args:
        grid (List[List[str]]): The word search grid to print
    """
    # Get the maximum width needed for each column (in case of varying character lengths)
    col_widths = [max(len(grid[row][col]) for row in range(len(grid))) for col in range(len(grid[0]))]
    
    # Print each row
    for row in grid:
        # Print each character in the row, padded to match the column width
        print(" ".join(char.ljust(width) for char, width in zip(row, col_widths)))


def print_found_matches(grid: List[List[str]], matches: List[List[Tuple[int, int]]]) -> None:
    """
    Print the grid with all found matches highlighted using coordinates.
    
    Args:
        grid (List[List[str]]): The word search grid
        matches (List[List[Tuple[int, int]]]): List of matches with their coordinates
    """
    if not matches:
        print("No matches found!")
        return
        
    print(f"Found {len(matches)} match(es):")
    
    for i, match in enumerate(matches, 1):
        print(f"\nMatch {i} coordinates:")
        for row, col in match:
            print(f"({row}, {col})", end=" ")
        print()

def string_to_grid(grid_string: str) -> List[List[str]]:
    """
    Convert a string-formatted grid (with spaces between letters) into a list of lists.
    
    Args:
        grid_string (str): The grid as a string, with spaces between letters and newlines between rows
        
    Returns:
        List[List[str]]: The grid as a list of lists
        
    Example:
        Input string:
        H E L L O
        W O R L D
        H E Y H I
        
        Output: [['H', 'E', 'L', 'L', 'O'], ['W', 'O', 'R', 'L', 'D'], ['H', 'E', 'Y', 'H', 'I']]
    """
    # Split the string into rows and remove any empty lines
    rows = [row.strip() for row in grid_string.strip().split('\n') if row.strip()]
    
    # Convert each row into a list of characters, splitting on whitespace
    grid = [row.split() for row in rows]
    
    # Validate that all rows have the same length
    if not all(len(row) == len(grid[0]) for row in grid):
        raise ValueError("All rows must have the same number of letters")
        
    return grid

# Example usage
if __name__ == "__main__":
    # Example grid
    grid = [
        ['O', 'R', 'U', 'O', 'F'],
        ['N', 'F', 'O', 'X', 'O'],
        ['E', 'W', 'O', 'X', 'U'],
        ['T', 'W', 'O', 'U', 'R'],
        ['X', 'F', 'O', 'U', 'R']
    ]
    
    # Example words to search
    test_words = ["ZERO", "ONE", "TWO", "FOUR"]
    
    print("Word Search Grid:")
    print_grid(grid)
    print()
    
    for word in test_words:
        print(f"\nSearching for '{word}':")
        matches = find_word(grid, word)

        print(matches)

        print_found_matches(grid, matches)
        