import customtkinter as ctk
import tkinter as tk
import joblib
import pandas as pd
import numpy as np

ctk.set_appearance_mode('Scolumnstem')
ctk.set_default_color_theme('green')

appWidth, appHeight = 1100, 600

heart_prediction = joblib.load('my_ml_model_eiei.joblib')

ui_font_big = ('Cascadia Code', 22, 'bold')
ui_font_small = ('Cascadia Code', 18, 'bold')

class App(ctk.CTk):

    def __init__(self, fg_color: str | None = None, **kwargs):
        super().__init__(fg_color, **kwargs)

        self.title('Tesla Model Linear')
        self.geometry(f'{appWidth}x{appHeight}')

        self.nameTitle = ctk.CTkLabel(
            self, text='Heart Disease Prediction', font=ui_font_big)
        self.nameTitle.grid(row=0, column=2, padx=10, pady=10)

        #age
        self.ageLabel = ctk.CTkLabel(
            self, text='Age', font=ui_font_small)
        self.ageLabel.grid(row=1, column=0, padx=10, pady=10)
        self.ageEntry = ctk.CTkEntry(self,
                                      placeholder_text="Enter age krub")
        self.ageEntry.grid(row=2, column=0, padx=5, pady=5)
        
        #gender label
        self.genderLabel = ctk.CTkLabel(self, text='Gender', font=ui_font_small)
        self.genderLabel.grid(row=3, column=0, padx=10, pady=10)
        #gender button
        self.genderVar = tk.StringVar(value='None')

        self.maleButton = ctk.CTkRadioButton(
            self, text='Male', variable=self.genderVar, value=1)
        self.maleButton.grid(row=3, column=1, padx=10, pady=10)

        self.femaleButton = ctk.CTkRadioButton(
            self, text='Female', variable=self.genderVar, value=0)
        self.femaleButton.grid(row=3, column=2, padx=10, pady=10)

        # #Chest pain type
        self.Chest_painTypeLabel = ctk.CTkLabel(self, text='Chest pain type', font=ui_font_small)
        self.Chest_painTypeLabel.grid(row=4, column=0, padx=10, pady=10)
        # #Chest pain type button
        self.ChestPainVar = tk.StringVar(value='None')

        self.CP_Level0 = ctk.CTkRadioButton(
            self, text='CP level : 0', variable=self.ChestPainVar, value=0)
        self.CP_Level0.grid(row=4, column=1, padx=10, pady=10)

        self.CP_Level1 = ctk.CTkRadioButton(
            self, text='CP level : 1', variable=self.ChestPainVar, value=1)
        self.CP_Level1.grid(row=4, column=2, padx=10, pady=10)

        self.CP_Level2 = ctk.CTkRadioButton(
            self, text='CP level : 2', variable=self.ChestPainVar, value=2)
        self.CP_Level2.grid(row=4, column=3, padx=10, pady=10)

        self.CP_Level3 = ctk.CTkRadioButton(
            self, text='CP level : 3', variable=self.ChestPainVar, value=3)
        self.CP_Level3.grid(row=4, column=4, padx=10, pady=10)

        #Resting blood pressure
        self.RBLabel = ctk.CTkLabel(
            self, text='Resting blood pressure', font=ui_font_small)
        self.RBLabel.grid(row=5, column=0, padx=10, pady=10)
        self.RBEntry = ctk.CTkEntry(self,
                                      placeholder_text="Resting blood pressure")
        self.RBEntry.grid(row=6, column=0, padx=10, pady=10)

        self.cholLabel = ctk.CTkLabel(self, text='Cholesterol', font=ui_font_small)
        self.cholLabel.grid(row=7,column=0, padx=10, pady=10)
        self.cholEntry = ctk.CTkEntry(self,
                                      placeholder_text="Cholesterol")
        self.cholEntry.grid(row=8,column=0, padx=10, pady=10)

        self.fbsLabel = ctk.CTkLabel(self, text='FBS', font=ui_font_small)
        self.fbsLabel.grid(row=7, column=1, padx=10, pady=10)
        self.fbsEntry = ctk.CTkEntry(self,
                                      placeholder_text="FBS")
        self.fbsEntry.grid(row=8, column=1, padx=10, pady=10)

        self.reLabel = ctk.CTkLabel(self, text='Resting ECG', font=ui_font_small)
        self.reLabel.grid(row=7, column=2, padx=10, pady=10)
        self.reEntry = ctk.CTkEntry(self,
                                      placeholder_text="RECG")
        self.reEntry.grid(row=8, column=2, padx=10, pady=10)

        self.thalachLabel = ctk.CTkLabel(self, text='Max Heart Rate By Electric', font=ui_font_small)
        self.thalachLabel.grid(row=7, column=3, padx=10, pady=10)
        self.thalachEntry = ctk.CTkEntry(self,
                                      placeholder_text="Heart Rate")
        self.thalachEntry.grid(row=8, column=3, padx=10, pady=10)

        self.exagLabel = ctk.CTkLabel(self, text='Is chest pain by ex?', font=ui_font_small)
        self.exagLabel.grid(row=1, column=2)
        self.exagCheckBox = ctk.CTkCheckBox(self, text='')
        self.exagCheckBox.grid(row=2, column=2)

        self.oldpeakLabel = ctk.CTkLabel(self, text="Oldpeak", font=ui_font_small)
        self.oldpeakLabel.grid(row=9, column=0, padx=10, pady=10)
        self.oldpeakEntry = ctk.CTkEntry(self,
                                      placeholder_text="Oldpeak")
        self.oldpeakEntry.grid(row=10, column=0, padx=10, pady=10)
        
        self.slopeLabel = ctk.CTkLabel(self, text="Slope", font=ui_font_small)
        self.slopeLabel.grid(row=9, column=1, padx=10, pady=10)
        self.slopeEntry = ctk.CTkEntry(self,
                                      placeholder_text="Slope")
        self.slopeEntry.grid(row=10, column=1, padx=10, pady=10)

        self.caLabel = ctk.CTkLabel(self, text="Major Vessels", font=ui_font_small)
        self.caLabel.grid(row=9, column=2, padx=10, pady=10)
        self.caEntry = ctk.CTkEntry(self,
                                      placeholder_text="Major Vessels")
        self.caEntry.grid(row=10, column=2, padx=10, pady=10)

        self.thalLabel = ctk.CTkLabel(self, text="Thal", font=ui_font_small)
        self.thalLabel.grid(row=9, column=3, padx=10, pady=10)
        self.thalEntry = ctk.CTkEntry(self,
                                      placeholder_text="Thal")
        self.thalEntry.grid(row=10, column=3, padx=10, pady=10)
        
        # BUTTON + PREDICTION
        self.predictButton = ctk.CTkButton(self, text='Predict', command=self.predict)
        self.predictButton.grid(row=11, column=2, padx=10, pady=10)
        self.predictLabel = ctk.CTkLabel(self, text='Prediction', font=ui_font_big)
        
    def predict(self):
        data = {}
        try:
            data = {
                "age": int(self.ageEntry.get()),
                "sex": int(self.genderVar.get()),
                "cp": int(self.ChestPainVar.get()),
                "trestbps": int(self.RBEntry.get()),
                "chol": int(self.cholEntry.get()),
                "fbs": int(self.fbsEntry.get()),
                "restecg": int(self.reEntry.get()),
                "thalach": int(self.thalachEntry.get()),
                "exang": int(self.exagCheckBox.get()),
                "oldpeak": float(self.oldpeakEntry.get()),
                "slope": int(self.slopeEntry.get()),
                "ca": int(self.caEntry.get()),
                "thal": int(self.thalEntry.get())
            }
        except:
            tk.messagebox.showerror(title="ERROR", message="Please enter valid value!")
        df = pd.DataFrame([data])
        result = heart_prediction.predict(df)
        if list(result) == [1]:
            tk.messagebox.showinfo(title="PREDICTION COMPLETE", message="You have a chance of heart disease.")
        else:
            tk.messagebox.showinfo(title="PREDICTION COMPLETE", message="You don't have a chance of heart disease.")
        self.ageEntry.delete(0, 'end')
        self.RBEntry.delete(0, 'end')
        self.cholEntry.delete(0, 'end')
        self.fbsEntry.delete(0, 'end')
        self.reEntry.delete(0, 'end')
        self.thalachEntry.delete(0, 'end')
        self.oldpeakEntry.delete(0, 'end')
        self.slopeEntry.delete(0, 'end')
        self.caEntry.delete(0, 'end')
        self.thalEntry.delete(0, 'end')


if __name__ == "__main__":
    app = App()
    app.mainloop()
