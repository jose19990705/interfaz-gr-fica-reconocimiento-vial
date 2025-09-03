# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 19:16:29 2025

@author: Jose Henao
"""

from tkinter import *

raiz= Tk()


MyFrame= Frame(raiz,width=500,height=400)
MyFrame.pack()

cuadroTexto= Entry(MyFrame)
cuadroTexto.place(x=100,y=100)
label=Label(MyFrame,text="Hola mundooooo",fg="blue")
label.place(x=200,y=30)
raiz.mainloop()