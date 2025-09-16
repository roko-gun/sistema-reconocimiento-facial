import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import pickle

class ReconocimientoFacialApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Reconocimiento Facial")
        self.root.geometry("900x600")
        self.root.configure(bg='#2c3e50')
        
        # Variables de control
        self.capturando = False
        self.reconociendo = False
        self.cap = None
        
        # Cargar el clasificador de rostros
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Directorios
        self.data_dir = "data_faces"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # Modelo de reconocimiento manual
        self.modelo_entrenado = False
        self.etiquetas = []
        self.rostros = []
        self.nombres = {}
        
        # Crear interfaz
        self.crear_interfaz()

    def crear_interfaz(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Título
        titulo = ttk.Label(main_frame, text="Sistema de Reconocimiento Facial", 
                          font=("Arial", 16, "bold"), foreground="#3498db")
        titulo.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Panel de video
        video_frame = ttk.LabelFrame(main_frame, text="Vista en tiempo real", padding="5")
        video_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        self.lbl_video = ttk.Label(video_frame, text="Cámara no iniciada", background="#34495e", 
                                  foreground="#ecf0f1", anchor="center")
        self.lbl_video.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Panel de controles
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Botones
        btn_capturar = ttk.Button(controls_frame, text="Capturar Rostros", command=self.iniciar_captura)
        btn_capturar.grid(row=0, column=0, padx=5)
        
        btn_entrenar = ttk.Button(controls_frame, text="Entrenar Modelo", command=self.entrenar_modelo)
        btn_entrenar.grid(row=0, column=1, padx=5)
        
        btn_reconocer = ttk.Button(controls_frame, text="Iniciar Reconocimiento", command=self.iniciar_reconocimiento)
        btn_reconocer.grid(row=0, column=2, padx=5)
        
        btn_detener = ttk.Button(controls_frame, text="Detener", command=self.detener_todo)
        btn_detener.grid(row=0, column=3, padx=5)
        
        # Entrada de nombre de usuario
        user_frame = ttk.Frame(controls_frame)
        user_frame.grid(row=1, column=0, columnspan=4, pady=10)
        
        ttk.Label(user_frame, text="Nombre:").grid(row=0, column=0, padx=5)
        self.nombre_var = tk.StringVar()
        entry_nombre = ttk.Entry(user_frame, textvariable=self.nombre_var, width=20)
        entry_nombre.grid(row=0, column=1, padx=5)
        
        # Panel de información
        info_frame = ttk.LabelFrame(main_frame, text="Información", padding="5")
        info_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        self.info_text = tk.Text(info_frame, height=8, width=70, state=tk.DISABLED)
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        
        # Estado
        self.status_var = tk.StringVar(value="Listo")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        
    def agregar_mensaje(self, mensaje):
        self.info_text.configure(state=tk.NORMAL)
        self.info_text.insert(tk.END, mensaje + "\n")
        self.info_text.see(tk.END)
        self.info_text.configure(state=tk.DISABLED)
        
    def iniciar_captura(self):
        if self.capturando or self.reconociendo:
            return
            
        nombre = self.nombre_var.get().strip()
        if not nombre:
            messagebox.showerror("Error", "Por favor, ingresa un nombre para la persona")
            return
            
        self.capturando = True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se puede acceder a la cámara")
            self.capturando = False
            return
            
        # Crear directorio para el usuario
        user_dir = os.path.join(self.data_dir, nombre)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            
        self.agregar_mensaje(f"Iniciando captura para: {nombre}")
        self.status_var.set(f"Capturando rostros de {nombre} - Sonría a la cámara")
        
        # Iniciar hilo para captura
        thread = threading.Thread(target=self.capturar_rostros, args=(user_dir,))
        thread.daemon = True
        thread.start()
        
    def capturar_rostros(self, user_dir):
        count = 0
        max_capturas = 30
        
        while self.capturando and count < max_capturas:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                rostro = gray[y:y+h, x:x+w]
                
                # Guardar imagen
                if count < max_capturas:
                    img_path = os.path.join(user_dir, f"{count}.jpg")
                    cv2.imwrite(img_path, rostro)
                    count += 1
                    
                # Mostrar contador
                cv2.putText(frame, f"Capturando: {count}/{max_capturas}", (10, 30), 
                           self.font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Mostrar video
            self.mostrar_frame(frame)
            time.sleep(0.1)
            
        self.cap.release()
        self.capturando = False
        self.agregar_mensaje(f"Captura completada. Se guardaron {count} imágenes")
        self.status_var.set("Captura completada")
        
    def entrenar_modelo(self):
        if self.capturando or self.reconociendo:
            return
            
        self.agregar_mensaje("Iniciando entrenamiento del modelo...")
        self.status_var.set("Entrenando modelo...")
        
        thread = threading.Thread(target=self.proceso_entrenamiento)
        thread.daemon = True
        thread.start()
        
    def proceso_entrenamiento(self):
        self.rostros = []
        self.etiquetas = []
        self.nombres = {}
        id_count = 0
        
        # Recorrer directorios
        for user_name in os.listdir(self.data_dir):
            user_dir = os.path.join(self.data_dir, user_name)
            if os.path.isdir(user_dir):
                self.nombres[id_count] = user_name
                for img_name in os.listdir(user_dir):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(user_dir, img_name)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (100, 100))
                            self.rostros.append(img)
                            self.etiquetas.append(id_count)
                id_count += 1
                
        if len(self.rostros) == 0:
            self.agregar_mensaje("Error: No hay imágenes para entrenar.")
            self.status_var.set("Error: No hay imágenes")
            return
            
        # Guardar modelo
        datos_modelo = {
            'rostros': self.rostros,
            'etiquetas': self.etiquetas,
            'nombres': self.nombres
        }
        
        with open("modelo_entrenado.pkl", "wb") as f:
            pickle.dump(datos_modelo, f)
            
        self.modelo_entrenado = True
        self.agregar_mensaje(f"Modelo entrenado con {len(self.rostros)} imágenes de {id_count} personas")
        self.status_var.set("Modelo entrenado correctamente")
        
    def cargar_modelo(self):
        if os.path.exists("modelo_entrenado.pkl"):
            with open("modelo_entrenado.pkl", "rb") as f:
                datos_modelo = pickle.load(f)
                self.rostros = datos_modelo['rostros']
                self.etiquetas = datos_modelo['etiquetas']
                self.nombres = datos_modelo['nombres']
                self.modelo_entrenado = True
                return True
        return False
        
    def reconocer_rostro(self, rostro):
        if not self.modelo_entrenado:
            return "Desconocido", 0
            
        rostro = cv2.resize(rostro, (100, 100))
        mejor_distancia = float('inf')
        mejor_id = -1
        
        for i, rostro_entrenado in enumerate(self.rostros):
            diferencia = cv2.absdiff(rostro, rostro_entrenado)
            distancia = np.mean(diferencia)
            
            if distancia < mejor_distancia:
                mejor_distancia = distancia
                mejor_id = self.etiquetas[i]
        
        if mejor_distancia < 50:
            confianza = max(0, 100 - mejor_distancia)
            return self.nombres.get(mejor_id, "Desconocido"), confianza
        else:
            return "Desconocido", 0
        
    def iniciar_reconocimiento(self):
        if self.capturando or self.reconociendo:
            return
            
        if not self.cargar_modelo():
            messagebox.showerror("Error", "Primero debe entrenar el modelo")
            return
            
        self.reconociendo = True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se puede acceder a la cámara")
            self.reconociendo = False
            return
            
        self.agregar_mensaje("Iniciando reconocimiento facial...")
        self.status_var.set("Reconociendo...")
        
        thread = threading.Thread(target=self.proceso_reconocimiento)
        thread.daemon = True
        thread.start()
        
    def proceso_reconocimiento(self):
        while self.reconociendo:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                rostro = gray[y:y+h, x:x+w]
                
                nombre, confianza = self.reconocer_rostro(rostro)
                
                if nombre != "Desconocido":
                    text = f"{nombre} ({int(confianza)}%)"
                    color = (0, 255, 0)
                else:
                    text = "Desconocido"
                    color = (0, 0, 255)
                    
                cv2.putText(frame, text, (x, y-10), self.font, 0.9, color, 2, cv2.LINE_AA)
            
            self.mostrar_frame(frame)
            
        self.cap.release()
        
    def mostrar_frame(self, frame):
        frame = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.lbl_video.configure(image=imgtk)
        self.lbl_video.image = imgtk
        
    def detener_todo(self):
        self.capturando = False
        self.reconociendo = False
        if self.cap is not None:
            self.cap.release()
        self.status_var.set("Detenido")
        self.agregar_mensaje("Proceso detenido")

# Código principal fuera de la clase
if __name__ == "__main__":
    root = tk.Tk()
    app = ReconocimientoFacialApp(root)
    root.mainloop()