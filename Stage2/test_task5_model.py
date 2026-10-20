import unittest
from datetime import date, time

from Stage2.task5_model import (
    CATEGORICAL_FEATURES,
    build_encoded_input,
    load_resources,
    transform_input,
)


class Task5ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resources = load_resources()

    def test_raw_input_is_encoded_in_saved_scaler_order(self) -> None:
        raw_data = self.resources["raw_data"]
        categories = {
            column: str(raw_data[column].dropna().mode().iloc[0])
            for column in CATEGORICAL_FEATURES
        }
        encoded = build_encoded_input(
            duration_minutes=120,
            show_date=date(2026, 10, 20),
            show_time=time(19, 30),
            release_date=date(2026, 9, 20),
            categories=categories,
            encoded_columns=self.resources["encoded_columns"],
        )

        self.assertEqual(
            list(encoded.columns),
            self.resources["encoded_columns"],
        )
        self.assertEqual(encoded.shape, (1, len(self.resources["encoded_columns"])))
        self.assertEqual(encoded.loc[0, "DurationMinutes"], 120)
        self.assertEqual(encoded.loc[0, "DaysFromReleaseToShow"], 30)

    def test_input_transforms_and_predicts_known_cluster(self) -> None:
        raw_data = self.resources["raw_data"]
        categories = {
            column: str(raw_data[column].dropna().mode().iloc[0])
            for column in CATEGORICAL_FEATURES
        }
        encoded = build_encoded_input(
            duration_minutes=120,
            show_date=date(2026, 10, 20),
            show_time=time(19, 30),
            release_date=date(2026, 9, 20),
            categories=categories,
            encoded_columns=self.resources["encoded_columns"],
        )
        pca_input = transform_input(
            encoded,
            self.resources["scaler"],
            self.resources["pca"],
        )

        prediction = self.resources["models"]["Decision Tree"].predict(pca_input)[0]
        self.assertEqual(pca_input.shape, (1, self.resources["features"].shape[1]))
        self.assertIn(prediction, self.resources["labels"].unique())


if __name__ == "__main__":
    unittest.main()
