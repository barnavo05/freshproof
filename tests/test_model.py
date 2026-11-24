import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from freshproof.models.classifier import MultiHeadClassifier  # noqa: E402


def test_output_shapes():
    model = MultiHeadClassifier(num_produce=5, num_grades=4)
    model.eval()
    with torch.no_grad():
        produce, grade = model(torch.randn(2, 3, 64, 64))
    assert produce.shape == (2, 5)
    assert grade.shape == (2, 4)


def test_rejects_empty_heads():
    with pytest.raises(ValueError):
        MultiHeadClassifier(num_produce=0, num_grades=4)
