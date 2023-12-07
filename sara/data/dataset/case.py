import os
from problog.logic import Term, Constant, Var, term2str
from problog.program import PrologString

class Case:
    def __init__(self,
            name: str,
            text: str,
            facts: PrologString,
            raw_facts: str,
            split: str = None,
            query: PrologString = None,
            question: str = None):
        self.name = name
        self.text = text
        self.facts = facts
        self.raw_facts = raw_facts
        self.split = split
        self.query = query
        self.question = question

    @classmethod
    def from_file(cls, filename):
        # take basename, remove file extension
        case_name = os.path.basename(filename)
        if case_name.endswith('.pl'):
            case_name = case_name[:-3]
        lines = [line.strip('\n') for line in open(filename, 'r')]
        # keep the text, and the knowledge graph
        text_index = lines.index('% Text')
        question_index = lines.index('% Question')
        facts_index = lines.index('% Facts')
        test_index = lines.index('% Test')
        halt_index = lines.index(':- halt.')
        case_text = Case.format_lines(lines[text_index+1:question_index])
        raw_case_facts = '\n'.join(lines[facts_index+1:test_index]) # for now, no formatting of facts at all
        case_facts = filter(lambda x: 'discontiguous' not in x, lines[facts_index+1:test_index])
        case_facts = PrologString('\n'.join(case_facts))
        query = PrologString('\n'.join(lines[test_index+1:halt_index]))
        question = Case.format_lines(lines[question_index+1:facts_index])
        return cls(case_name, case_text, case_facts, raw_case_facts, query=query, question=question)

    @classmethod
    def format_lines(cls, lines):
        """ used to format textual description of a case, and a question """
        text = lines
        text = [line.lstrip('% ').rstrip(' ') for line in text]
        text = list(filter(lambda x: len(x)>0, text))
        text = ' '.join(text)
        return text

