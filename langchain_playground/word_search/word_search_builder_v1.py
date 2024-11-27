from langchain_core.prompts import PromptTemplate
from prompts.v1_AgentPuzzlemaker_prompt import v1_AgentPuzzlemaker_prompt
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.output_parsers.string import StrOutputParser


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
    def __init__(self):
        pass

    def make_puzzle(self, model, words, n_rows, n_cols, prompt=None):
        
        if prompt is None:
            prompt = v1_AgentPuzzlemaker_prompt

        chain = prompt | model | StrOutputParser()

        return chain.invoke({
            'words': ', '.join(words),
            'n_rows': n_rows,
            'n_cols': n_cols
        })

class AgentWordSearch(object):
    '''
    Example: 

    from dotenv import load_dotenv
    load_dotenv()

    from langchain_anthropic import ChatAnthropic
    model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=.5)

    from word_search_builder_v1 import AgentWordSearch
    ws = AgentWordSearch()
    ws.create_word_search(model, 'math', 5, 7, 7)

    print(ws.words)
    print(ws.puzzle)
    '''
    def __init__(self):
        self.words = None
        self.puzzle = None

    def create_word_search(self, model, topic, n_words, n_rows, n_cols, puzzlemaker_prompt=None):
        words = AgentWordpicker().pick_words(model, topic, n_words, n_rows, n_cols)
        puzzle = AgentPuzzlemaker().make_puzzle(model, words, n_rows, n_cols, prompt=puzzlemaker_prompt)

        self.words = [w.upper() for w in words]
        self.puzzle = puzzle

        pass

    def print(self):
        pass