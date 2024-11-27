from langchain_core.prompts import PromptTemplate

v1_AgentPuzzlemaker_prompt = PromptTemplate(
    template = """
    INSTRUCTIONS:
    You are an expert word search puzzle creator.  
    Create a word search using {words}.
    The word search should be {n_rows} rows and {n_cols} columns.
    Each word should appear once and only once.
    Each word should be in a straight line in the puzzle, either horizontally, vertically, or diagonally.
    Words should not pivot within the puzzle.
    Words should not have extra letters inserted.

    EXAMPLE 1:
    Words are dog, cat, ant
    Puzzle is 5 rows and 5 columns
    Example puzzle is
    X X X X D
    X T X X O
    T N X X G
    X A X X X
    X X C X X

    EXAMPLE 2:
    Words are mouse, beetle, bear
    Puzzle is 8 rows and 8 columns
    Example puzzle is
    X X X X X X X X
    X X X B X X X X
    X X X E X X X X
    X X X E S X X X
    X X X T X U X X
    X X X L X X O X
    X R A E B X X M
    X X X X X X X X

    INCORRECT EXAMPLE:
    Words are add, sum, minus, divide, ratio
    Puzzle is 7 rows and 7 columns
    Incorrect example puzzle is:
    A D D I V I D
    R A T I X X E
    X X X O X X M
    S U M X X X I
    X X I X X X N
    X X N X X X U
    M I N U S X S

    OUTPUT:
    Return only the letter grid with one space between letters on each row.  Do not provide any other output such as "Here's a 7X7 word search puzzle grind containing the requested words:"
    """,
    input_variables = ['words', 'n_rows', 'n_cols']
)