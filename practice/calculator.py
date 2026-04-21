# 개선 코드
def calculate(num1:float, num2: float, op: str) -> float:
    match op:
        case "+":
            return num1 + num2
        case "-":
            return num1 - num2
        case "*":
            return num1 * num2
        case "/":
            if num2 == 0:
                raise ZeroDivisionError("0으로 나눌 수 없습니다.")
            return num1 / num2
        case _:
            raise ValueError(f"지원하지 않는 연산자입니다.")


def main() -> None:
    while True:
        op = input("연산자 (+, -, *, /, q): ")
        if op == 'q':
            print("프로그램을 종료합니다.")
            break

        num1 = float(input("숫자 1: "))
        num2 = float(input("숫자 2: "))
        try:
            result = calculate(num1, num2, op)
            print(f"{num1} {op} {num2} = {result}")
        except (ZeroDivisionError, ValueError) as e:
            print(f"오류: {e}")


if __name__ == "__main__":
    main()