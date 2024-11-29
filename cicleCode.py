import numpy as np
import math
from itertools import combinations, zip_longest
import operator


def get_error(remainder):
    if remainder == [0, 0, 1]:
        return [0, 0, 0, 0, 0, 0, 1]
    elif remainder == [0, 1, 0]:
        return [0, 0, 0, 0, 0, 1, 0]
    elif remainder == [1, 0, 0]:
        return [0, 0, 0, 0, 1, 0, 0]
    elif remainder == [0, 1, 1]:
        return [0, 0, 0, 1, 0, 0, 0]
    elif remainder == [1, 1, 0]:
        return [0, 0, 1, 0, 0, 0, 0]
    elif remainder == [1, 1, 1]:
        return [0, 1, 0, 0, 0, 0, 0]
    elif remainder == [1, 0, 1]:
        return [1, 0, 0, 0, 0, 0, 0]


class CyclicEncoder:
    def __init__(self, n, k):
        # self.generator_polynomial = np.array([1, 0, 1, 1])  # x^3 + x + 1
        self.n = n
        self.k = k

        if self.n == 7:
            self.generator_polynomial = np.array([1, 0, 1, 1])
        elif self.n == 15:
            self.generator_polynomial = np.array([1, 0, 0, 1, 1])

    def encode(self, message):
        if len(message) != self.k:
            raise ValueError(f"Сообщение должно быть длиной {self.k} бит")

        message_array = np.array(message, dtype=int)

        padded_message = np.pad(message_array, (0, self.n - self.k))

        _, remainder = np.polydiv(padded_message, self.generator_polynomial)

        remainder = np.pad(remainder, (self.n - self.k - len(remainder), 0))

        remainder_bits = remainder.astype(int) % 2

        codeword = padded_message.copy()

        codeword[-(self.n - self.k):] ^= remainder_bits

        return codeword.tolist(), remainder_bits.tolist()


def get_all_error_vectors(length: int, mult: int):
    all_vectors = []
    if mult > length:
        return all_vectors  # Невозможно создать комбинацию с более единицами, чем длина вектора

    # Получаем все возможные комбинации позиций для mult единиц
    positions = list(combinations(range(length), mult))

    # Для каждой комбинации позиций создаем вектор ошибки
    for pos in positions:
        error_vector = ['0'] * length
        # Устанавливаем '1' в выбранных позициях
        for p in pos:
            error_vector[p] = '1'
        all_vectors.append(''.join(error_vector))

    return all_vectors


def mod_2_add(pol1: list, pol2: str):
    pol2_new = list(pol2)
    pol2_new = list(map(int, pol2_new))
    # print(f'pol1 : {pol1}')
    # print(f'pol2_new : {pol2_new}')
    summed_list = [x ^ y for x, y in zip(pol1, pol2_new)]
    # print(f'summed_list : {summed_list}')
    return summed_list


def main():
    n, k = map(int, input().split())
    vect_for_coding = list(map(str, input().split()))
    encoder = CyclicEncoder(n, k)
    print(f'Информационный вектор : {vect_for_coding}')
    coded_vect, start_remainder = encoder.encode(vect_for_coding)
    print(f'Закодированный вектор : {coded_vect}')
    # print(coded_vect)

    detectedErrors = [0] * (n + 1)
    corrections = [0] * (n + 1)
    total_errors = 2 ** n - 1
    percent_per_error_group = [0] * (n+1)
    errors_per_group = [0] * (n+1)

    for mult in range(1, n+1):
        cnt = 0
        current_mult_errors = math.comb(n, mult)
        errors_per_group[mult] = current_mult_errors

        error_vectors = get_all_error_vectors(7, mult)
        for error_vect in error_vectors:
            coded_vect_copy = coded_vect.copy()
            coded_vect_plus_error = mod_2_add(coded_vect.copy(), error_vect)
            coded_vect_plus_error_copy = coded_vect_plus_error.copy()
            coded_vect_plus_error_pol = np.array(coded_vect_plus_error)
            generator_pol = np.array(encoder.generator_polynomial)
            quotient, remainder = np.polydiv(coded_vect_plus_error_pol, generator_pol)
            remainder = np.pad(remainder, (n - k - len(remainder), 0))
            remainder_bits = remainder.astype(int) % 2
            remainder_bits = remainder_bits.tolist()
            if 1 in remainder_bits:
                error = get_error(remainder_bits)
                for ind, i in enumerate(error):
                    if i == 1:
                        if coded_vect_plus_error_copy[ind] == 1:

                            coded_vect_plus_error_copy[ind] = 0
                        else:
                            coded_vect_plus_error_copy[ind] = 1
                        coded_vect_plus_error_copy_pol = np.array(coded_vect_plus_error_copy)
                        quotient_, remainder_ = np.polydiv(coded_vect_plus_error_copy_pol, generator_pol)
                        remainder_ = np.pad(remainder_, (n - k - len(remainder_), 0))
                        remainder_bits_ = remainder_.astype(int) % 2
                        remainder_bits_ = remainder_bits_.tolist()
                        if 1 not in remainder_bits_:
                            pos_corrected = coded_vect_plus_error_copy.copy()
                            for index, num in enumerate(start_remainder):
                                pos_corrected[len(pos_corrected) - index - 1] = (pos_corrected[len(pos_corrected) - index - 1] + start_remainder[len(start_remainder) - 1 - index])%2

                            pos_corrected = pos_corrected[:k]
                            pos_corrected = list(map(str, pos_corrected))
                            if pos_corrected == vect_for_coding:
                                corrections[mult] += 1
                        else:
                            print(error)


        from tabulate import tabulate
    

        headers = ["Кратность ошибки", "Число ошибок данной кратности", "Исправлено ошибок",
                   "Корректирующая способность"]
    data = []

    for i in range(1, n + 1):
        data.append([i, errors_per_group[i], corrections[i], corrections[i] / errors_per_group[i]])

    print(tabulate(data, headers=headers, floatfmt=".2f"))


main()
