from abc import ABC, abstractmethod
import argparse

from get_data import get_dataset, get_dataset_split


class Parser(ABC):
    def __init__(self):
        self.train = get_dataset_split("train")
        self.val = get_dataset_split("validation")
        self.test = get_dataset_split("test")
        self._train()

    @abstractmethod
    def _train(self):
        pass

    def predict(self):
        correct_unlabeled, correct_labeled = 0, 0
        num_tokens = 0
        for i, sentence in enumerate(self.test):
            sentence_copy = sentence.copy()
            sentence_copy["head"] = [] * len(sentence["tokens"])
            sentence_copy["deprel"] = [] * len(sentence["tokens"])
            sentence_copy["deps"] = [] * len(sentence["tokens"])
            heads, deprels = self.parse_sentence(sentence_copy)
            sent_correct_unlabeled, sent_correct_labeled = Parser.get_correct(sentence, heads, deprels)
            correct_unlabeled += sent_correct_unlabeled
            correct_labeled += sent_correct_labeled
            num_tokens += len(sentence["tokens"])
            # if i > 10:
            #     break

        return correct_unlabeled / num_tokens, correct_labeled / num_tokens, num_tokens

    @abstractmethod
    def parse_sentence(self, sentence):
        pass

    @classmethod
    def get_correct(self, sentence, heads, deprels):
        assert len(sentence["tokens"]) == len(heads) == len(deprels)
        assert len([h for h in heads if h == 0]) == 1, f"{sentence['tokens']}\n" \
                                                       f"{sentence['xpos']}\n" \
                                                       f"{heads}"

        correct_unlabeled, correct_labeled = 0, 0
        for g_head, g_deprel, p_head, p_deprel in zip(sentence["head"], sentence["deprel"], heads, deprels):
            if g_head == str(p_head):
                correct_unlabeled += 1
                if g_deprel == p_deprel:
                    correct_labeled += 1
        return correct_unlabeled, correct_labeled


class DummyParser(Parser):
    def _train(self):
        # This is a dummy parser: no training
        pass

    def parse_sentence(self, sentence):
        heads = [-1] * len(sentence["tokens"])
        deprels = ["unk"] * len(sentence["tokens"])
        verb_found = False
        for i, pos in enumerate(sentence["xpos"]):
            if pos is None:
                continue
            if pos.startswith("V"):
                heads[i] = 0
                for j in range(len(sentence["tokens"])):
                    if i != j:
                        heads[j] = i+1
                verb_found = True
                break
        if not verb_found:
            heads[0] = 0
        return heads, deprels


class LessDummyParser(Parser):
    def _train(self):
        # This is a less dummy parser: no training
        pass

    def parse_sentence(self, sentence):
        heads = [-1] * len(sentence["tokens"])
        heads[0] = 0
        deprels = ["unk"] * len(sentence["tokens"])
        # YOUR CODE GOES HERE
        # Write rules that are better than the ones in DummyParser
        return heads, deprels


class ArcStandardParser(Parser):
    def _train(self):
        # YOUR CODE GOES HERE
        # Use self.train to "learn" an oracle for the Arc Standard parsing algorithm
        #   Don't use "serious" machine learning. A simple method to get the best operator given a configuration
        #     is enough
        #   The "majority" oracle could just choose the most frequent transition (shift, right_arc, left_arc)
        #     given the part-of-speech tag of the top-2 elements of the stack
        #     (and perhaps also the first element in the buffer)
        self.oracle = None
        pass

    def parse_sentence(self, sentence):
        # YOUR CODE GOES HERE
        # Write code that uses the oracle to parse the sentence
        heads = [-1] * len(sentence["tokens"])
        heads[0] = 0
        deprels = ["unk"] * len(sentence["tokens"])
        return heads, deprels


if __name__ == "__main__":
    for parser in DummyParser, LessDummyParser, ArcStandardParser:
        parser = parser()
        uas, las, num_tokens = parser.predict()
        print(f"{parser.__class__.__name__:20} Unlabeled Accuracy (UAS): {uas:.3f} [{num_tokens} tokens]")
        print(f"{parser.__class__.__name__:20} Labeled Accuracy (UAS):   {las:.3f} [{num_tokens} tokens]")
        print()
