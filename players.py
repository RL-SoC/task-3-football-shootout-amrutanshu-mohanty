class player():
    x: int
    y : int
    
    def __init__(self, x : int, y : int) -> None:
        self.x = x
        self.y = y
    
    def move(self, direction : int):
        if direction == 0:
            self.y -= 1
        elif direction == 1:
            self.x += 1
        elif direction == 2:
            self.y += 1
        elif direction == 3:
            self.x -= 1

class attacker(player):
    def __init__(self, x: int, y: int) -> None:
        super().__init__(x, y)
    