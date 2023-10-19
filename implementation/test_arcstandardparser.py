from parser import ArcStandardParser

parser = ArcStandardParser()
uas, las, num_tokens = parser.predict()


def test_hmm_20():
    assert uas >= 0.20


def test_hmm_25():
    assert uas >= 0.25


def test_hmm_30():
    assert uas >= 0.30


def test_hmm_35():
    assert uas >= 0.35


def test_hmm_40():
    assert uas >= 0.40


def test_hmm_45():
    assert uas >= 0.45


def test_hmm_50():
    assert uas >= 0.50


def test_hmm_55():
    assert uas >= 0.55


def test_hmm_60():
    assert uas >= 0.60


def test_hmm_65():
    assert uas >= 0.65


def test_hmm_70():
    assert uas >= 0.70


def test_hmm_75():
    assert uas >= 0.75


def test_hmm_80():
    assert uas >= 0.80
