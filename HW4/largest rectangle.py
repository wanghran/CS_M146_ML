def rec(hist):
    stack = []
    prev = -1
    i = 0
    largest = 0
    while i < len(hist):
        if len(stack) == 0:
            stack.append((i, hist[i]))
            prev = 0
            i += 1
        elif stack[prev] is not None and hist[i] > stack[prev][1]:
            stack.append((i, hist[i]))
            prev += 1
            i += 1
        else:
            while len(stack) is not 0 and stack[prev][1] > hist[i]:
                area = stack[prev][1] * (i - prev)
                largest = max(largest, area)
                stack.pop()
                if prev == 0:
                    prev = prev
                else:
                    prev = prev - 1
    while len(stack) is not 0:
        area = stack[prev][1] * (i - stack[prev][0])
        largest = max(largest, area)
        stack.pop()
        if prev == 0:
            prev = prev
        else:
            prev -= 1

    return largest


h = [1, 4, 5, 3, 2]
print(rec(h))