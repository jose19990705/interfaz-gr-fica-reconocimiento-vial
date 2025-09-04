from backend import PavementProcessor
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de pavimento")
        self.root.geometry("900x650")  # ventana más compacta
        self.ruta_video = None
        self.ruta_salida = None

        # Variables
        self.var_min_inicio = tk.StringVar(value="0")
        self.var_min_fin = tk.StringVar(value="0")
        self.var_todo = tk.BooleanVar(value=True)
        self.var_estado = tk.StringVar(value="Listo")

        # Construcción interfaz
        self.crear_interfaz()

    def crear_interfaz(self):
        # --- Parte superior (video de inferencia) ---
        frame_video = tk.Frame(self.root, bg="skyblue", bd=3, relief="groove")
        frame_video.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(
            frame_video,
            text="Detección de irregularidad",
            font=("Arial", 12, "bold"),
            bg="skyblue",
        ).pack(pady=5)

        self.label_inferencia = tk.Label(
            frame_video, text="(espacio detección)", bg="white"
        )
        self.label_inferencia.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Parte inferior (controles) ---
        frame_inferior = tk.Frame(self.root)
        frame_inferior.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # Botones
        btn_cargar = tk.Button(
            frame_inferior, text="Cargar video", width=15, command=self.cargar_video
        )
        btn_cargar.grid(row=0, column=0, columnspan=2, pady=5)

        btn_guardar = tk.Button(
            frame_inferior, text="Guardar salida", width=15, command=self.guardar_salida
        )
        btn_guardar.grid(row=1, column=0, columnspan=2, pady=5)

        tk.Label(frame_inferior, text="Modelo: YOLOv8").grid(
            row=2, column=0, columnspan=2, pady=5
        )

        tk.Label(frame_inferior, text="Min inicio:").grid(row=3, column=0, sticky="e")
        tk.Entry(frame_inferior, textvariable=self.var_min_inicio, width=6).grid(
            row=3, column=1, sticky="w"
        )

        tk.Label(frame_inferior, text="Min fin:").grid(row=4, column=0, sticky="e")
        tk.Entry(frame_inferior, textvariable=self.var_min_fin, width=6).grid(
            row=4, column=1, sticky="w"
        )

        tk.Checkbutton(
            frame_inferior, text="Todo el video", variable=self.var_todo
        ).grid(row=5, column=0, columnspan=2, pady=5)

        btn_iniciar = tk.Button(
            frame_inferior, text="Iniciar inferencia", width=18, command=self.iniciar
        )
        btn_iniciar.grid(row=6, column=0, columnspan=2, pady=10)

        # Estado
        self.label_estado = tk.Label(
            frame_inferior, textvariable=self.var_estado, fg="blue", anchor="center"
        )
        self.label_estado.grid(row=7, column=0, columnspan=2, pady=5, sticky="we")

        # Barra de progreso
        self.progress = ttk.Progressbar(
            frame_inferior, orient="horizontal", length=300, mode="determinate"
        )
        self.progress.grid(row=8, column=0, columnspan=2, pady=10)

    def cargar_video(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[
                ("Archivos de video", "*.mp4 *.avi *.mov"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if ruta:
            self.ruta_video = ruta
            self.var_estado.set(f"Video cargado: {ruta}")
        else:
            self.var_estado.set("No se seleccionó video")

    def guardar_salida(self):
        ruta = filedialog.asksaveasfilename(
            title="Guardar video procesado",
            defaultextension=".mp4",
            filetypes=[
                ("Video MP4", "*.mp4"),
                ("Video AVI", "*.avi"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if ruta:
            self.ruta_salida = ruta
            self.var_estado.set(f"Salida: {ruta}")
        else:
            self.var_estado.set("No se seleccionó ruta de salida")

    def iniciar(self):
        if not self.ruta_video:
            messagebox.showwarning("Aviso", "Primero carga un video.")
            return
        if not self.ruta_salida:
            messagebox.showwarning("Aviso", "Selecciona la ruta de salida.")
            return

        ini = int(self.var_min_inicio.get())
        fin = int(self.var_min_fin.get())
        todo = self.var_todo.get()

        self.var_estado.set("Ejecutando inferencia...")
        self.progress["value"] = 0
        self.root.update_idletasks()

        def worker():
            procesador = PavementProcessor()
            salida = procesador.procesar_video(
                video_path=self.ruta_video,
                output_path=self.ruta_salida,
                inicio_min=ini,
                fin_min=fin,
                todo=todo,
                callback=self.mostrar_frame,
            )
            self.var_estado.set(f"✅ Video procesado en: {salida}")
            messagebox.showinfo("Finalizado", f"Video procesado en:\n{salida}")

        threading.Thread(target=worker, daemon=True).start()

    def mostrar_frame(self, frame_inferido, porcentaje):
        def _update():
            frame_rgb = cv2.cvtColor(frame_inferido, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Obtener tamaño actual del label
            ancho = self.label_inferencia.winfo_width()
            alto = self.label_inferencia.winfo_height()

            if ancho > 1 and alto > 1:
                img = img.resize((ancho, alto))  # ajustamos al espacio del label

            img_tk = ImageTk.PhotoImage(img)
            self.label_inferencia.configure(image=img_tk)
            self.label_inferencia.image = img_tk

            self.progress["value"] = porcentaje
            self.var_estado.set(f"Progreso: {porcentaje:.1f}%")

        self.root.after(0, _update)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
