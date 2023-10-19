import pprint

from datasets import load_dataset
# from datasets import get_dataset_split_names


def get_dataset_split(split):
    assert split in ["train", "validation", "test"]
    return load_dataset("universal_dependencies", "en_ewt", split=split)


def get_dataset():
    return get_dataset_split("train"), get_dataset_split("validation"), get_dataset_split("test")


def demo(ds):
    # https://huggingface.co/datasets/universal_dependencies
    print(ds)
    for sentence in ds:
        # pprint.pprint(sentence)
        for token, pos, head, deprel in zip(sentence["tokens"], sentence["xpos"],
                                            sentence["head"], sentence["deprel"]):
            print(f"{token:15} {pos:5} {head:2} {deprel:5}")
        break


if __name__ == "__main__":
    train, val, test = get_dataset()
    demo(train)
