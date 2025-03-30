def print_triangle(height):
    for i in range(height):
        # 打印空格
        for j in range(height - i - 1):
            print(" ", end="")
        # 打印星号
        for k in range(2 * i + 1):
            print("*", end="")
        print()  # 换行

# 设置三角形高度
height = 5
print("打印一个三角形:")
print_triangle(height)

