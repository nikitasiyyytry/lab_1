"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = {"a", "e", "i", "o", "u"}
    return sum(character.lower() in vowels for character in text)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return number.bit_count()


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if len(predicted) != len(expected):
        raise ValueError("Vectors must have the same length")
    if len(predicted) == 0:
        raise ValueError("Vectors must not be empty")

    squared_error = sum(
        (predicted_value - expected_value) ** 2
        for predicted_value, expected_value in zip(predicted, expected)
    )
    return squared_error / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors: list[str] = []
    divisor = 2

    while divisor * divisor <= number:
        power = 0
        while number % divisor == 0:
            number //= divisor
            power += 1
        if power:
            factors.append(f"({divisor})" if power == 1 else f"({divisor}**{power})")
        divisor = 3 if divisor == 2 else divisor + 2

    if number > 1:
        factors.append(f"({number})")
    return "".join(factors)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    level = 0
    used_cubes = 0
    while used_cubes < cube_count:
        level += 1
        used_cubes += level**2
    return level if used_cubes == cube_count else "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = str(number)
    side_length = (len(digits) - 1) // 2
    return sum(map(int, digits[:side_length])) == sum(map(int, digits[-side_length:])) if side_length else True
