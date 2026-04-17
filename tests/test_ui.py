from src.ui.app import PredictorUI


class _DummyPredictor:
    def predict_next(self, text: str, k: int):
        return ["holmes", "watson", "lestrade"][:k]


def test_get_predictions_returns_list_of_strings():
    ui = PredictorUI(_DummyPredictor())
    result = ui.get_predictions("holmes looked", 2)
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


def test_get_predictions_handles_empty_input():
    ui = PredictorUI(_DummyPredictor())
    assert ui.get_predictions("", 3) == []
