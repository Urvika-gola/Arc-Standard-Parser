from parser import LessDummyParser

parser = LessDummyParser()
uas, las, num_tokens = parser.predict()


def test_hmm_15():
    assert uas >= 0.15


def test_hmm_20():
    assert uas >= 0.20


def test_hmm_25():
    assert uas >= 0.25