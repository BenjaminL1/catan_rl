import random


class StackedDice:
    def __init__(self):
        self.bag = []
        self._refill_bag()

    def _refill_bag(self):
        # Create a list of the 36 standard 2-dice sums
        # 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1
        distribution = {
            2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
            8: 5, 9: 4, 10: 3, 11: 2, 12: 1
        }
        self.bag = []
        for num, count in distribution.items():
            self.bag.extend([num] * count)

        # Add Noise: Randomly remove 1 non-7 value and replace it with a random number (2-12)
        non_sevens = [x for x in self.bag if x != 7]
        if non_sevens:
            to_remove = random.choice(non_sevens)
            self.bag.remove(to_remove)
            self.bag.append(random.randint(2, 12))

        random.shuffle(self.bag)

    def roll(self, current_player_obj, last_7_roller_obj):
        # Karma Check: If last_7_roller_obj is NOT None AND last_7_roller_obj != current_player_obj
        if last_7_roller_obj is not None and last_7_roller_obj != current_player_obj:
            # Roll a d100 (0-99). If < 20 (20% chance), return 7 immediately.
            if random.randint(0, 99) < 20:
                return 7

        # Refill Check
        if not self.bag:
            self._refill_bag()

        # Standard Roll: Pop the last value from self.bag
        return self.bag.pop()
