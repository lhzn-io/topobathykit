import numpy as np

from topobathykit.quality import QualityClass, TIDClassifier, source_report


def test_classifier_direct() -> None:
    assert TIDClassifier.classify(11) == QualityClass.DIRECT
    assert TIDClassifier.classify(17) == QualityClass.DIRECT
    assert TIDClassifier.get_color(QualityClass.DIRECT) == "green"


def test_classifier_indirect() -> None:
    assert TIDClassifier.classify(40) == QualityClass.INDIRECT
    assert TIDClassifier.get_color(QualityClass.INDIRECT) == "orange"


def test_classifier_unknown() -> None:
    assert TIDClassifier.classify(0) == QualityClass.UNKNOWN
    assert TIDClassifier.classify(99) == QualityClass.UNKNOWN  # Unlisted defaults to unknown
    assert TIDClassifier.get_color(QualityClass.UNKNOWN) == "grey"


def test_source_report() -> None:
    # Create sample data: 5 Direct, 3 Indirect, 2 Unknown = 10 total
    data = np.array(
        [
            11,
            11,
            11,
            11,
            11,  # 5 Direct
            40,
            40,
            40,  # 3 Indirect
            0,
            50,  # 2 Unknown
        ]
    )

    report = source_report(data)

    assert report[QualityClass.DIRECT.value] == 50.0
    assert report[QualityClass.INDIRECT.value] == 30.0
    assert report[QualityClass.UNKNOWN.value] == 20.0


def test_source_report_empty() -> None:
    report = source_report(np.array([]))
    assert report[QualityClass.DIRECT.value] == 0.0
