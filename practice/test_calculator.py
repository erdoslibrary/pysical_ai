from calculator import calculate

test_cases = [
    (10, 3, "+", 13.0),
    (10, 3, "-", 7.0),
    (10, 3, "*", 30.0),
    (10, 3, "/", 10 / 3),
]

print("=== 정상 케이스 ===")
for num1, num2, op, expected in test_cases:
    result = calculate(num1, num2, op)
    status = "PASS" if result == expected else "FAIL"
    print(f"[{status}] {num1} {op} {num2} = {result}")

print("\n=== 예외 케이스 ===")

try:
    calculate(10, 0, "/")
except ZeroDivisionError as e:
    print(f"[PASS] 0 나누기 예외: {e}")

try:
    calculate(10, 3, "%")
except ValueError as e:
    print(f"[PASS] 잘못된 연산자 예외: {e}")
