import asyncio
import unittest

import main
from main import normalize_recipe_result


class RecipeNutritionNormalizationTest(unittest.TestCase):
    def test_dry_pasta_uses_pre_cooking_weight_without_yield_multiplier(self):
        data = {
            "name": "Makaron z tunczykiem",
            "total_weight_grams": 2200,
            "ingredient_details": [
                {
                    "line": "500 g makaronu suchego",
                    "name": "makaron pszenny",
                    "amount_grams": 500,
                    "state": "dry",
                    "calories_per_100g": 350,
                    "protein_per_100g": 12,
                    "carbs_per_100g": 72,
                    "sugar_per_100g": 3,
                    "fat_per_100g": 1.5,
                    "saturated_fat_per_100g": 0.3,
                    "fiber_per_100g": 3,
                    "salt_per_100g": 0.02,
                },
                {
                    "line": "240 g tunczyka odsaczonego w sosie wlasnym",
                    "name": "tunczyk w sosie wlasnym",
                    "amount_grams": 240,
                    "state": "drained",
                    "calories_per_100g": 116,
                    "protein_per_100g": 26,
                    "carbs_per_100g": 0,
                    "sugar_per_100g": 0,
                    "fat_per_100g": 1,
                    "saturated_fat_per_100g": 0.3,
                    "fiber_per_100g": 0,
                    "salt_per_100g": 1,
                },
                {
                    "line": "300 g pomidorow",
                    "name": "pomidory",
                    "amount_grams": 300,
                    "state": "raw",
                    "calories_per_100g": 18,
                    "protein_per_100g": 0.9,
                    "carbs_per_100g": 3.9,
                    "sugar_per_100g": 2.6,
                    "fat_per_100g": 0.2,
                    "saturated_fat_per_100g": 0.03,
                    "fiber_per_100g": 1.2,
                    "salt_per_100g": 0.01,
                },
                {
                    "line": "1 lyzka oliwy (opcjonalnie)",
                    "name": "oliwa",
                    "amount_grams": 14,
                    "state": "as_sold",
                    "included": False,
                    "calories_per_100g": 884,
                    "protein_per_100g": 0,
                    "carbs_per_100g": 0,
                    "sugar_per_100g": 0,
                    "fat_per_100g": 100,
                    "fiber_per_100g": 0,
                    "salt_per_100g": 0,
                },
            ],
        }

        result = normalize_recipe_result(data, servings=5)

        self.assertEqual(result["servings"], 5)
        self.assertEqual(result["ingredients_weight_grams"], 1040)
        self.assertEqual(result["total_weight_grams"], 2200)
        self.assertEqual(result["fat"], 11)
        self.assertEqual(result["saturated_fat"], 2.3)
        self.assertEqual(result["calories"], 2082)
        self.assertEqual(result["nutrition_calculation_method"], "ingredient_sum_v1")
        self.assertIn("suchego", result["ingredients"][0])
        self.assertIn("opcjonalnie", result["ingredients"][-1])

    def test_legacy_response_without_breakdown_keeps_ai_totals(self):
        result = normalize_recipe_result(
            {
                "name": "Starszy przepis",
                "calories": 1200,
                "fat": 45,
                "ingredients": ["500 g makaronu"],
            },
            servings=4,
        )

        self.assertEqual(result["calories"], 1200)
        self.assertEqual(result["fat"], 45)
        self.assertNotIn("nutrition_calculation_method", result)

    def test_description_only_analysis_does_not_require_placeholder_image(self):
        original_generate = main.generate_text_json
        main.generate_text_json = lambda _: {
            "name": "Makaron z tunczykiem",
            "ingredients": ["300 g makaronu suchego"],
            "steps": ["Ugotuj makaron."],
            "ingredient_details": [
                {
                    "line": "300 g makaronu suchego",
                    "name": "makaron",
                    "amount_grams": 300,
                    "state": "dry",
                    "calories_per_100g": 350,
                    "protein_per_100g": 12,
                    "carbs_per_100g": 72,
                    "fat_per_100g": 1.5,
                }
            ],
            "total_weight_grams": 750,
        }
        try:
            result = asyncio.run(
                main.analyze_recipe(
                    file=None,
                    description="300 g makaronu suchego",
                    servings=3,
                    meal_type="Obiad",
                    ai_provider="gemini",
                )
            )
        finally:
            main.generate_text_json = original_generate

        self.assertEqual(result["servings"], 3)
        self.assertEqual(result["fat"], 5)
        self.assertEqual(result["total_weight_grams"], 750)
        self.assertEqual(result["nutrition_calculation_method"], "ingredient_sum_v1")


if __name__ == "__main__":
    unittest.main()
