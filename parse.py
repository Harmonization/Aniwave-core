import re

import numpy as np
from sympy import lambdify, symbols
from sympy.parsing.sympy_parser import parse_expr
from sympy.printing.latex import LatexPrinter

DATA_PATH = 'Data'

class Parser:
    # Парсинг математических выражений из строки
    
    def __init__(self):
        self.latex_printer = LatexPrinter()

    def __call__(self, string):
        self.str2expr(string)
        self.str2nm(string)

    def str2expr(self, string: str) -> None:
        # Получение функции из строки
        
        # str -> sympy-expression
        expr = parse_expr(string)

        # Поиск переменных (роль которых выполняют номера каналов, заданные в виде 'СловоЧисло', напр. 'b70' или 'канал70')
        lets = symbols(list(set(re.findall(r'[a-zA-Zа-яА-Я]+\d+', string))), integer=True, positive=True) # переменные

        # sympy-expression -> функция python
        self._function = lambdify(lets, expr)

        # переменные -> номера каналов
        self._bands: list[int] = [int(re.search(r'\d+', str(let))[0]) for let in lets]
        
    def str2nm(self, string: str) -> None:
        # Трансформация строки: номера каналов заменяются на нанометры (таблица соответствий хранится в wave)

        def transform_string(string: str) -> str:
            el = string.group(0)
            return f"{re.search(r'[a-zA-Zа-яА-Я]+', el)[0]}{self.wave[int(re.search(r'[0-9]+', el)[0])]}"
        
        # str_band -> str_nm
        string_nm = re.sub(r'[a-zA-Zа-яА-Я]+\d+', transform_string, string)

        # str_nm -> sympy-expression
        expr_nm = parse_expr(string_nm)

        # sympy-expression -> формула latex
        self._latex_nm = self.latex_printer.doprint(expr_nm)

    def channel(self, hsi: np.ndarray[float]) -> np.ndarray[float]:
        # Вычислить одноканальное изображение в соответствии с полученной функцией
        return self._function(*[hsi[:, :, b] for b in self.bands])

    @property
    def latex(self):
        return self._latex_nm
    
    @property
    def function(self):
        return self._function
    
    @property
    def bands(self):
        return self._bands
    
    @property
    def wave(self) -> np.ndarray[int]:
        if not hasattr(self, '_wave'):
            self._wave = np.load(f'{DATA_PATH}/nm.npy')
        return self._wave
    
    @property
    def latex_nm(self):
        return self._latex_nm

def main():
    string = 'b70 + b60 / 3'
    parser = Parser()
    parser(string)
    print(f'latex_nm: {parser.latex}')

    string = 'b70/2 + b50 / 3'
    parser(string)
    print(f'latex_nm: {parser.latex}')


if __name__ == '__main__':
    main()