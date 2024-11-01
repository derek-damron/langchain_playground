from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.output_parsers.string import StrOutputParser


#####
# Setup
#

class AgentWordpicker(object):
    def __init__(self):
        pass

    def pick_words(self, model, topic, n_words, n_rows, n_cols):
        output_parser = CommaSeparatedListOutputParser()
        format_instructions = output_parser.get_format_instructions

        prompt = PromptTemplate(
            template = """
            You are an expert on {topic}.
            List {n_words} words about {topic} that are most than 2 letters long and less than {max_word_length} letters long.
            {format_instructions}
            """,
            input_variables = ['topic', 'n_words', 'max_word_length'],
            partial_variables = {'format_instructions': format_instructions}
        )

        chain = prompt | model | output_parser

        return chain.invoke({
            'topic': topic,
            'n_words': n_words,
            'max_word_length': min(n_rows, n_cols)
        })

class AgentPuzzlemaker(object):
    '''
    Example: 

    from dotenv import load_dotenv
    load_dotenv()

    from langchain_anthropic import ChatAnthropic
    model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=.5)

    from word_search_builder_v1 import AgentWordsearch
    ws = AgentWordsearch().create_word_search(model, 'math', 5, 7, 7)

    print(ws['words'])
    print(ws['puzzle'])
    '''
    def __init__(self):
        pass

    def make_puzzle(self, model, words, n_rows, n_cols):
        
        prompt = PromptTemplate(
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

        chain = prompt | model | StrOutputParser()

        return chain.invoke({
            'words': ', '.join(words),
            'n_rows': n_rows,
            'n_cols': n_cols
        })

class AgentWordsearch(object):
    def __init__(self):
        pass

    def create_word_search(self, model, topic, n_words, n_rows, n_cols):
        words = AgentWordpicker().pick_words(model, topic, n_words, n_rows, n_cols)
        puzzle = AgentPuzzlemaker().make_puzzle(model, words, n_rows, n_cols)

        return {
            'words': words,
            'puzzle': puzzle
        }