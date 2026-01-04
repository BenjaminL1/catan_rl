import collections


class ResourceTracker:
    def __init__(self, player_names):
        # Dictionary to store the 'Belief State' of every player's hand
        # Format: {'PlayerName': {'BRICK': 0, 'ORE': 0, ...}}
        self.knowledge = {}
        self.all_resources = ['BRICK', 'ORE', 'SHEEP', 'WHEAT', 'WOOD']

        for name in player_names:
            self.knowledge[name] = {r: 0 for r in self.all_resources}

    def track_gain(self, player_name, resource, amount=1):
        """Called when a player gains resources (Dice Roll, Year of Plenty)"""
        if player_name in self.knowledge and resource in self.knowledge[player_name]:
            self.knowledge[player_name][resource] += amount

    def track_loss(self, player_name, resource, amount=1):
        """Called when a player spends resources (Building)"""
        if player_name in self.knowledge and resource in self.knowledge[player_name]:
            # Ensure we don't go below zero (in case of sync errors)
            self.knowledge[player_name][resource] = max(
                0, self.knowledge[player_name][resource] - amount)

    def track_trade(self, p1_name, p1_gives, p1_amount, p2_name, p2_gives, p2_amount):
        """Called for Bank or Player trades"""
        # P1 gives resources
        self.track_loss(p1_name, p1_gives, p1_amount)
        # P1 gets resources
        self.track_gain(p1_name, p2_gives, p2_amount)

        # If P2 is a player (not 'BANK'), swap for them
        if p2_name != 'BANK':
            self.track_loss(p2_name, p2_gives, p2_amount)
            self.track_gain(p2_name, p1_gives, p1_amount)

    def track_steal(self, thief_name, victim_name, resource):
        """Called when Robber steals a specific card"""
        # In 1v1, the Thief knows what they got, so the Victim knows what they lost.
        # It is Public Information relative to the two players.
        if resource:
            self.track_gain(thief_name, resource, 1)
            self.track_loss(victim_name, resource, 1)

    def track_discard(self, player_name, resource_list):
        """Called when 7 is rolled and cards are dumped"""
        for res in resource_list:
            self.track_loss(player_name, res, 1)

    def track_initial_resources(self, player_name, resource_list):
        """Called when a player receives initial resources from their second settlement"""
        for res in resource_list:
            self.track_gain(player_name, res, 1)

    def get_observable_hand(self, player_name):
        """Returns the estimated resource counts for a player as a list"""
        # Order: BRICK, ORE, SHEEP, WHEAT, WOOD
        if player_name not in self.knowledge:
            return [0, 0, 0, 0, 0]
        return [self.knowledge[player_name][r] for r in self.all_resources]
