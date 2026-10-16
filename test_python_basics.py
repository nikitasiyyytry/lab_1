import unittest

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput
from python_basics import (
    count_one_bits,
    count_vowels,
    has_unique_characters,
    is_balanced_number,
    mse,
    multiplicative_persistence,
    prime_factorization,
    pyramid,
)


class PythonBasicsTests(unittest.TestCase):
    def test_count_vowels_is_case_insensitive(self):
        self.assertEqual(count_vowels(TextInput("EdUcAtIon")), 5)
        self.assertEqual(count_vowels(TextInput("rhythm")), 0)

    def test_unique_characters(self):
        self.assertTrue(has_unique_characters(TextInput("")))
        self.assertTrue(has_unique_characters(TextInput("abcA")))
        self.assertFalse(has_unique_characters(TextInput("letter")))

    def test_count_one_bits(self):
        self.assertEqual(count_one_bits(PositiveIntegerInput(1)), 1)
        self.assertEqual(count_one_bits(PositiveIntegerInput(0b101101)), 4)

    def test_multiplicative_persistence(self):
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(4)), 0)
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(39)), 3)
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(999)), 4)

    def test_mse(self):
        self.assertEqual(mse(VectorPairInput([1, 2, 3], [1, 4, 5])), 8 / 3)
        self.assertEqual(mse(VectorPairInput([7], [7])), 0.0)

    def test_prime_factorization(self):
        self.assertEqual(prime_factorization(PositiveIntegerInput(86240)), "(2**5)(5)(7**2)(11)")
        self.assertEqual(prime_factorization(PositiveIntegerInput(13)), "(13)")
        self.assertEqual(prime_factorization(PositiveIntegerInput(1)), "")

    def test_pyramid(self):
        self.assertEqual(pyramid(PositiveIntegerInput(1)), 1)
        self.assertEqual(pyramid(PositiveIntegerInput(14)), 3)
        self.assertEqual(pyramid(PositiveIntegerInput(15)), "It is impossible")

    def test_balanced_number(self):
        self.assertTrue(is_balanced_number(PositiveIntegerInput(1234006)))
        self.assertFalse(is_balanced_number(PositiveIntegerInput(123456)))
        self.assertTrue(is_balanced_number(PositiveIntegerInput(124521)))
        self.assertTrue(is_balanced_number(PositiveIntegerInput(7)))


if __name__ == "__main__":
    unittest.main()
