from customtkinter import *

app = CTk()
app.geometry('500x400')
app.title('Model')
set_appearance_mode('light')

label = CTkLabel(app, text='Heart Disease Prediction')
label.pack()

app.mainloop()
