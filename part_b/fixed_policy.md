# Fixed policy for Part B (written BEFORE any experiment)

Map (state number = 4 * row + col):

```text
 0 S   1 F   2 F   3 F
 4 F   5 H   6 F   7 H
 8 F   9 F  10 F  11 H
12 H  13 F  14 F  15 G
```

Action codes of FrozenLake: 0 = left, 1 = down, 2 = right, 3 = up.

| State | Action | Reason |
| --- | --- | --- |
| 0 | down (1) | start of the safe route along the left edge; right/down are both safe, down avoids the hole row 1 |
| 4 | down (1) | goes to 8; 'right' would enter the hole at 5 |
| 8 | right (2) | 'down' would enter the hole at 12 |
| 9 | right (2) | 'down' (13) is safe but longer; 'up' is the hole 5 |
| 10 | down (1) | 'right' would enter the hole at 11; 14 is next to the goal |
| 14 | right (2) | the goal 15 |
| 1 | right (2) | not on the main route; keeps moving toward 2 |
| 2 | down (1) | toward 6 |
| 6 | down (1) | toward 10; 'right' (7) and 'left' (5) are holes |
| 3 | left (0) | 'down' (7) is a hole |
| 13 | right (2) | toward 14 |

Route without slipping: 0 -> 4 -> 8 -> 9 -> 10 -> 14 -> 15 (6 steps, no hole is touched).
The policy reacts only to the current state, is the same in every experiment, and is never changed after seeing results.
It is designed for the map and does not account for slipping, so I expect it to be much worse when `is_slippery=True`.
