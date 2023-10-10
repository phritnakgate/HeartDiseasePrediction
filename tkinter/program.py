import customtkinter as ctk
import tkinter as tk

ctk.set_appearance_mode('System')
ctk.set_default_color_theme('dark-blue')

appWidth, appHeight = 600, 600


class App(ctk.CTk):
    def __init__(self, fg_color: str | None = None, **kwargs):
        super().__init__(fg_color, **kwargs)

        self.title('Tesla Model Linear')
        self.geometry(f'{appWidth}x{appHeight}')

        self.nameTitle = ctk.CTkLabel(
            self, text='Heart Disease Prediction', font=('Cascadia Code', 22, 'bold'))
        self.nameTitle.place(x=150, y=20)

        self.nameLabel = ctk.CTkLabel(
            self, text='Name :', font=('Cascadia Code', 18, 'bold'))
        self.nameLabel.place(x=30, y=80)

        self.nameEntry = ctk.CTkEntry(self,
                                      placeholder_text="Enter name krub")
        self.nameEntry.place(x=150, y=80)


if __name__ == "__main__":
    app = App()
    app.mainloop()
