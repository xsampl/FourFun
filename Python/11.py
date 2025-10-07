import tkinter as tk

class AutoClickerGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Autoclicker Game")
        self.score = 0
        self.auto_clicks = 0
        self.auto_click_cost = 1
        self.auto_auto_clicks = 0
        self.auto_auto_click_cost = 1

        self.score_label = tk.Label(root, text=f"Score: {self.score}", font=("Arial", 16))
        self.score_label.pack(pady=10)

        self.click_button = tk.Button(root, text="Click Me!", font=("Arial", 16), command=self.manual_click)
        self.click_button.pack(pady=10)

        self.buy_auto_clicker_button = tk.Button(
            root,
            text=f"Buy Auto Clicker ({self.auto_click_cost} points)",
            font=("Arial", 12),
            command=self.buy_auto_clicker
        )

        self.buy_auto_auto_clicker_clicker_button = tk.Button(
            root,
            text=f"Buy Auto Auto Clicker Clicker ({self.auto_auto_click_cost} points)",
            font=("Arial", 12),
            command=self.buy_auto_auto_clicker
        )

        self.buy_auto_clicker_button.pack(pady=10)
        self.buy_auto_auto_clicker_clicker_button.pack(pady=10)

        self.auto_clicker_label = tk.Label(root, text=f"Auto Clickers: {self.auto_clicks}", font=("Arial", 12))
        self.auto_clicker_label.pack(pady=10)

        self.auto_auto_clicker_clicker_label = tk.Label(root, text=f"Auto Auto Clicker Clickers: {self.auto_auto_clicks}", font=("Arial", 12))
        self.auto_auto_clicker_clicker_label.pack(pady=10)

        self.root.after(1000, self.auto_click)

    def manual_click(self):
        self.score += 1
        self.update_labels()

    def buy_auto_clicker(self):
        if self.score >= self.auto_click_cost:
            self.score -= self.auto_click_cost
            self.auto_clicks += 1
            self.auto_click_cost = int(self.auto_click_cost * 2)
            self.buy_auto_clicker_button.config(text=f"Buy Auto Clicker ({self.auto_click_cost} points)")
            self.update_labels()

    def buy_auto_auto_clicker(self):
        if self.score >= self.auto_auto_click_cost:
            self.score -= self.auto_auto_click_cost
            self.auto_auto_clicks += 1
            self.auto_auto_click_cost = int(self.auto_auto_click_cost * 2)
            self.buy_auto_auto_clicker_clicker_button.config(text=f"Buy Auto Auto Clicker Clicker ({self.auto_auto_click_cost} points)")
            self.update_labels()
            self.root.after(1000, self.auto_auto_click)
    
    def auto_auto_click(self):
        self.buy_auto_clicker()
        self.update_labels()
        self.root.after(1000, self.auto_auto_click)

    def auto_click(self):
        self.score += self.auto_clicks
        self.update_labels()
        self.root.after(1000, self.auto_click)

    def update_labels(self):
        self.score_label.config(text=f"Score: {self.score}")
        self.auto_clicker_label.config(text=f"Auto Clickers: {self.auto_clicks}")
        self.auto_auto_clicker_clicker_label.config(text=f"Auto Auto Clicker Clickers: {self.auto_auto_clicks}")

if __name__ == "__main__":
    root = tk.Tk()
    game = AutoClickerGame(root)
    root.mainloop()