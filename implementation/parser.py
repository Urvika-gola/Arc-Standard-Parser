from abc import ABC, abstractmethod
import argparse
from collections import defaultdict
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

    def find_head(self, idx, tokens, xpos, root_index):
        if xpos[idx] is 'None':
            return None
        if xpos[idx].startswith("N") and xpos[idx - 1].startswith("V") and idx > 0 and xpos[idx - 1]:
            return idx - 1
        if xpos[idx].startswith("N") and xpos[idx + 1] and idx < len(tokens) - 1  and xpos[idx + 1].startswith("V"):
            if idx + 1 != root_index:
                return idx + 1
        if xpos[idx].startswith("RB") and idx < len(tokens) - 1 and xpos[idx + 1] and xpos[idx + 1].startswith("V"):
            return idx + 1
        if xpos[idx].startswith("J") and idx < len(tokens) - 1 and xpos[idx + 1] and xpos[idx + 1].startswith("N"):
            return idx + 1
        if tokens[idx] in ["with", "about", "at"] and idx > 0 and (
                xpos[idx - 1] and xpos[idx - 1].startswith("V") or xpos[idx - 1] and xpos[idx - 1].startswith("N")):
            return idx - 1
        if tokens[idx] == "'s" and idx > 0:
            return idx + 1
        if tokens[idx] in ["and", "or"] and idx > 0:
            return idx - 1
        # Handling two nouns together, apposition
        if idx > 1 and tokens[idx - 1] == "," and xpos[idx] and xpos[idx].startswith("N") and xpos[idx - 2] and xpos[
            idx - 2].startswith("N"):
            return idx - 2
        if idx < len(tokens) - 1 and idx + 1 != root_index:
            return idx + 1
        elif idx > 0 and idx - 1 != root_index:
            return idx - 1
        return root_index

    def parse_sentence(self, sentence):
        tokens = sentence["tokens"]
        deprels = ["unk"] * len(sentence["tokens"])
        root_index = 0
        root_node = []
        xpos = sentence["xpos"]
        heads = [-1] * len(sentence["tokens"])
        heads[0], heads[root_index]  = 0, 0
        for idx, token in enumerate(tokens):
            if idx != root_index:
                head = self.find_head(idx, tokens, xpos, root_index)
                if head is not None:
                    heads[idx] = head + 1
        return heads, deprels


class ArcStandardParser(Parser):
    def _train(self):
        self.shift_counts_words = defaultdict(int)
        self.left_arc_counts_words = defaultdict(int)
        self.right_arc_counts_words = defaultdict(int)
        self.shift_counts = defaultdict(int)
        self.left_arc_counts = defaultdict(int)
        self.right_arc_counts = defaultdict(int)
        for sentence in self.train:
            tokens = sentence["tokens"] # get the tokens from the sentence
            stack = [0]  # Root
            buffer = list(range(1, len(tokens) + 1))
            while buffer or len(stack) > 1:
                # Get the POS tags for the top 2 stack elements and the first buffer element
                s1 = stack[-1] if len(stack) > 0 else None
                s2 = stack[-2] if len(stack) > 1 else None
                b1 = buffer[0] if buffer else None
                s1_pos = sentence["xpos"][s1 - 1] if s1 else 'None'
                s2_pos = sentence["xpos"][s2 - 1] if s2 else 'None'
                b1_pos = sentence["xpos"][b1 - 1] if b1 else 'None'
                config = (s2_pos, s1_pos, b1_pos)
                if s2 and s1 and sentence["head"][s2 - 1] is not None and sentence["head"][
                    s2 - 1] != 'None' and s1 == int(sentence["head"][s2 - 1]):
                    self.left_arc_counts[config] = self.left_arc_counts.get(config, 0) + 1
                    stack.pop(-2)
                elif s2 and s1 and sentence["head"][s1 - 1] is not None and (
                        sentence["head"][s1 - 1] != 'None' and s2 == int(sentence["head"][s1 - 1])):
                    self.right_arc_counts[config] = self.right_arc_counts.get(config, 0) + 1
                    stack.pop()
                elif buffer:
                    self.shift_counts[config] = self.shift_counts.get(config, 0) + 1
                    stack.append(buffer.pop(0))
                else:
                    # If buffer is empty, we can't perform left or right arc, then break
                    break
                config_words = (
                tokens[s2 - 1] if s2 else None, tokens[s1 - 1] if s1 else None, tokens[b1 - 1] if b1 else None)
                if s2 and s1 and sentence["head"][s2 - 1] is not None and sentence["head"][
                    s2 - 1] != 'None' and s1 == int(sentence["head"][s2 - 1]):
                    self.left_arc_counts_words[config_words] += 1
                elif s2 and s1 and sentence["head"][s1 - 1] not in [None, 'None'] and s2 == int(
                        sentence["head"][s1 - 1]):
                    self.right_arc_counts_words[config_words] += 1
                elif buffer:
                    self.shift_counts_words[config_words] += 1
                if config not in self.shift_counts and config not in self.left_arc_counts and config not in self.right_arc_counts:
                    if len(stack) > 1:
                        self.right_arc_counts[config] = self.right_arc_counts.get(config, 0) + 1
                        stack.pop()
                    elif buffer:
                        self.shift_counts[config] = self.shift_counts.get(config, 0) + 1
                        stack.append(buffer.pop(0))

    def parse_sentence(self, sentence):
        tokens = sentence["tokens"]
        stack = [0]
        buffer = list(range(1, len(tokens) + 1))
        heads = ['None'] * len(tokens)  # <-- Initialize with None instead of -1
        deprels = ["unk"] * len(tokens)
        if len(tokens) == 1:
            return [0], ["root"]  # or any default deprel
        while buffer or len(stack) > 1:
            s1 = stack[-1] if len(stack) > 0 else None
            s2 = stack[-2] if len(stack) > 1 else None
            b1 = buffer[0] if buffer else None
            s1_pos = sentence["xpos"][s1 - 1] if s1 else 'None'
            s2_pos = sentence["xpos"][s2 - 1] if s2 else 'None'
            b1_pos = sentence["xpos"][b1 - 1] if b1 else 'None'
            config = (s2_pos, s1_pos, b1_pos)
            config_words = (
            tokens[s2 - 1] if s2 else None, tokens[s1 - 1] if s1 else None, tokens[b1 - 1] if b1 else None)
            shift_weight = self.shift_counts_words.get(config_words, 0)
            left_arc_weight = self.left_arc_counts_words.get(config_words, 0)
            right_arc_weight = self.right_arc_counts_words.get(config_words, 0)
            if shift_weight == 0 and left_arc_weight == 0 and right_arc_weight == 0:
                shift_weight = self.shift_counts.get(config, 0)
                left_arc_weight = self.left_arc_counts.get(config, 0)
                right_arc_weight = self.right_arc_counts.get(config, 0)
            if s2 and left_arc_weight >= shift_weight and left_arc_weight >= right_arc_weight:
                heads[s2 - 1] = s1
                deprels[s2 - 1] = 'LEFT-ARC'
                stack.pop(-2)
            elif s2 and right_arc_weight > shift_weight:
                heads[s1 - 1] = s2
                deprels[s1 - 1] = 'RIGHT-ARC'
                stack.pop()
            elif buffer:
                stack.append(buffer.pop(0))
            else:
                if len(stack) > 1:
                    heads[s2 - 1] = s1
                    deprels[s2 - 1] = 'LEFT-ARC'
                    stack.pop(-2)
        if 0 not in heads:
            priorities = ["V", "NNP", "NN"]
            validate_root = False
            for option in priorities:
                for i, pos in enumerate(sentence["xpos"]):
                    if pos and pos.startswith(option):
                        heads[i] = 0
                        validate_root = True
                        break
                if validate_root:
                    break
            if not validate_root:
                for i, pos in enumerate(sentence["xpos"]):
                    if pos not in ["-LRB-", "-RRB-", ",", ".", ":", "IN", "TO", "HYPH"]:
                        heads[i] = 0
                        break
        if sentence["xpos"][-1] in [".", ",", ":", ";", "'", "`"]:
            heads[-1] = heads.index(0) + 1
        # Loop over all 'None' POS tag indices and adjust relationships
        for i in range(len(sentence["tokens"])):
            if heads[i] == 'None' and sentence["xpos"][i] is not None:
                heads[i] = heads.index(0) + 1
        return heads, deprels


if __name__ == "__main__":
    for parser in DummyParser, LessDummyParser, ArcStandardParser:
        parser = parser()
        uas, las, num_tokens = parser.predict()
        print(f"{parser.__class__.__name__:20} Unlabeled Accuracy (UAS): {uas:.3f} [{num_tokens} tokens]")
        print(f"{parser.__class__.__name__:20} Labeled Accuracy (UAS):   {las:.3f} [{num_tokens} tokens]")
        print()
