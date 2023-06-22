#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
educhallenge.py

Authors: Jan Heysel & Johanna Rätz
Version: X2 (Stand: 13.06.2023 = zweite Version für den dritten Zyklus)

Diese Bibliothek enthält Funktionen, die in der Simulation der EduChallenge benötigt werden und selbst geschrieben wurden, die aber für Schüler:innen nicht sichtbar sein sollen, um die Oberfläche der Simulation übersichtlich zu halten. 

Neu in X2: es wird ein Download-Button angezeigt, über den man die erzeugten Dateien herunterladen kann. Außerdem wird ein Upload-Button angezeigt, um das Wurfvideo in den aktuellen Ordner hochzuladen. 
"""

# Bibliotheken
import cv2 # zum Einlesen des Videos -> nicht in Anaconda, aber in opencv
import numpy as np # diverse numerische Operationen
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # see https://matplotlib.org/stable/tutorials/introductory/images.html
import ipympl # -> nicht in Anaconda
from scipy.optimize import curve_fit
from numpy.polynomial import Chebyshev
import os
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from datetime import datetime
from ipywidgets import HTML
from IPython.display import display
import base64
import ipywidgets as widgets

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
#####                                                                   #####
##### Funktionen für die überarbeitete Version "ModellBildung im Flug"  #####
#####                                                                   #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

##### Downloadbutton #####

def DownloadButton(filename):
    res = 'computed results'
    #FILE
    b64 = base64.b64encode(res.encode())
    payload = b64.decode()

    #BUTTONS
    html_buttons = '''<html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>
    <body>
    <a href="{filename}" download>
    <button class="p-Widget jupyter-widgets jupyter-button widget-button mod-warning">Download Datei</button>
    </a>
    </body>
    </html>
    '''

    html_button = html_buttons.format(payload=payload,filename=filename)
    display(HTML(html_button))
    
   
    
##### Uploadbutton (schreibt Dateiname in Datei) #####
     
def Button_um_Video_hochzuladen():
    upload_button = widgets.FileUpload(description='Video Upload')
    display(upload_button)

    uploaded_filename = None  # Variable to store the uploaded filename

    def handle_upload_button_change(change):
        global uploaded_filename
        uploaded_file = upload_button.value
        if uploaded_file:
            file_name = list(uploaded_file.keys())[0]
            file_contents = uploaded_file[file_name]['content']
            # Save the uploaded file to the notebook's directory
            file_path = os.path.join(os.getcwd(), file_name)
            with open(file_path, 'wb') as file:
                file.write(file_contents)
            uploaded_filename = file_name
            print(f"Die Datei '{file_name}' wurde gespeichert unter: {file_path}")
            
            f = open("Dateinamen_Wurfvideo.txt", "w")
            f.write(uploaded_filename)
            f.close()
            
    upload_button.observe(handle_upload_button_change, names='value')
    


##### Zelle Vorbereitung #####

def vorbereitungen(B):
    
    # Ordner für die Bilder anlegen, wenn es ihn noch nicht gibt.
    try:
        os.mkdir("Ausgangsbilder")
    except:
        unnuetz = 42
    
    # Video in Bilder zerlegen
    N_Bilder, fps = video_in_bilder_zerlegen()
    
    # reale Bildhöhe H bestimmen: 
    bild1 = mpimg.imread("Ausgangsbilder/frame00.jpg")
    h, b, layers = bild1.shape # Höhe h und Breite b des Videos in Pixeln
    H = B*h/b # die Höhe H des Bildausschnitts in Metern können wir nun berechnen (wegen H/B = h/b)
    
    # Ausgabe, damit man sieht, dass es erfolgreich war.
    print("Informationen und Video erfolgreich eingelesen.")
    print("Der Bildausschnitt im Realvideo ist " + str(np.round(B, 3)) + " m breit und " + str(np.round(H, 3)) + " m hoch.")
    
    return N_Bilder, fps, h, b, H



def video_in_bilder_zerlegen():
    """
    Diese Funktion speichert die im Video enthaltenen Bilder als einzelne .jpg-Bilder und gibt N_Bilder, die Zahl der gespeicherten Bilder und fps, die Frames pro Sekunde, zurück.
    """
    
    # Den Dateinamen des Realvideos einlesen:
    f = open("Dateinamen_Wurfvideo.txt", "r")
    dateiname_realvideo = f.read()
    
    
    vidcap = cv2.VideoCapture(dateiname_realvideo)

    # Ausgabe der einzelnen Bilder des Realvideos als einzelne jpg-Dateien | speichern im gerade erstellten Unterordner:
    success,image = vidcap.read()
    #print(success)
    count = 0
    while success:
      cv2.imwrite("Ausgangsbilder" + "/frame" + str(format(count, '02d')) + ".jpg", image)     # save frame as JPEG file 
      #print("Bild " + str(format(count, '02d')) + ".jpg" + " gespeichert.")
      success,image = vidcap.read()
      count += 1
    N_Bilder = count # +1 (glaube ich falsch)
    
    # get fps:
    # see https://learnopencv.com/how-to-find-frame-rate-or-frames-per-second-fps-in-opencv-python-cpp/

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.
    if int(major_ver)  < 3 :
        fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
        #print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        #print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    
    return N_Bilder, fps



##### Zelle Videoanalyse #####

# siehe: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.connect.html   
# die Funktion "onclick" muss in das Skript selber, sonst funktioniert es nicht
def define_figure(B, H):
    """ Diese Funktion dient dazu für die Lernenden verwirrenden, aber für den Ablauf notwendigen Code zu verstecken."""
    
    # passe die Bildbreite so an, dass sie gut auf den Bildschirm passt (größere Länge von Breite B, Höhe H, sollte Dim = "10" sein):
    Dim = 10 # im Idealfall würde Dim an die Bildschirmgröße angepasst. s. https://stackoverflow.com/questions/3129322/how-do-i-get-monitor-resolution-in-python
    if B > H: # Querformat
        skalierung = Dim / B
    else: # Hochformat oder Quadrat
        skalierung = Dim / H
    
    fig, ax = plt.subplots(figsize=(B*skalierung, H*skalierung))
    ax.set_xlim([0, B])
    ax.set_ylim([0, H])
    ax.set_xlabel("x-Position / m")
    ax.set_ylabel("y-Position / m")
    
    return fig, ax

def define_fmes(N_Bilder, B, H):
    """Erstellt eine Liste fmes und fügt alle Bilder hinzu."""
    
    fmes = []
    
    for i in range(N_Bilder):
        p = "Ausgangsbilder/frame" + str(format(i, '02d')) + ".jpg"          # relativer Dateipfad und Name des Bildes
        img = mpimg.imread(p)                                                # das Bild geladen
        fmes.append([plt.imshow(img, extent=[0, B, 0, H], animated=True)]) # das Bild der Liste hinzugefügt
        fmes.append([plt.imshow(img, extent=[0, B, 0, H], animated=True)]) # aus unbekannten Gründen wird nur jedes zweite Bild der Liste angezeigt. Wenn die Bilder doppelt drin sind, ist dies effektiv jedes Bild...
        
    return fmes



##### Zelle Messwerte anzeigen und speichern #####

# Vorteil im Vergleich zum zweiten Zyklus: weil wir das Skript zur Erfassung der Punkte angepasst haben, liegen die Arrays x und y bereits 
# in der Einheit Meter vor. Das Umrechnen entfällt. 

def Messwerte_zeigen_und_speichern(x, y, B, H, b, h, fps, N_Bilder):
    
    # einen Plot ausgeben und speichern, damit man die Bahnkurve schon mal sieht:
    fig, ax = plt.subplots()
    plt.plot(x, y, "bo")
    ax.set_title("Beobachtete Flugbahn des Objekts \n Messwerte aus Antippen im Video")
    ax.set_xlabel("x-Position [m]")
    ax.set_ylabel("y-Position [m]")
    plt.show()
    plt.savefig("Messwerte_aus_Antippen.png")  
    
    # Bestimme N_Punkte. Weil es sein kann, dass ein Bild nicht oder doppelt angeklickt wurde, ist das nicht notwendig = N_Bilder (wenn es eigentlich auch so sein sollte).
    N_Punkte = len(x)
    
    # Zeit der Bilder bestimmen: 
    delta_t = 1/fps # Zeit zwischen zwei Bildern
    zeitpunkte = np.arange(0, N_Punkte*delta_t, delta_t)
    
    # Schätzer für die Startwerte berechnen und ausgeben:
    x_0, y_0, v_x0, v_y0 = berechne_startwerte_fit(fps, x, y)
    rundung = 2
    print_Schaetzer_Anfangswerte(x_0, y_0, v_x0, v_y0, rundung)
    
    # Alle Werte speichern. Als Exceltabelle speichern, damit Jugendliche auch außerhalb des Projekt damit weiterarbeiten könnten    
    dict_Videoanalyse_positionen = {"Zeitpunkte der Bilder/Positionen in s seit erstem Bild": zeitpunkte, "x-Positionen in m": x, "y-Positionen in m": y}

    dict_Videoanalyse_parameter = {"Bildbreite B (real) in m": B, "Bildhoehe H (real) in m": H, "Bildbreite B (in Pixeln)": b, "Bildhoehe H (in Pixeln)": h, 
                        "fps": fps, "delta t in s": delta_t, "Anzahl Bilder": N_Bilder, "Anzahl Punkte": N_Punkte, 
                        "Schätzer x_0 in m": x_0, "Schätzer y_0 in m": y_0, "Schätzer v_x0 in m/s": v_x0, "Schätzer v_y0 in m/s": v_y0}

    Videoanalyse_Positionen = pd.DataFrame.from_dict(dict_Videoanalyse_positionen)
    Videoanalyse_Parameter  = pd.DataFrame.from_dict(dict_Videoanalyse_parameter, orient='index', columns = ["Werte"])

    Videoanalyse_Positionen.to_excel("Videoanalyse_Positionen.xlsx")
    Videoanalyse_Parameter.to_excel("Videoanalyse_Parameter.xlsx")
    
    return N_Punkte



def berechne_startwerte_fit(fps, x_m, y_m):
    """
    Berechnet die Startwerte auf Basis eines Fits über die empirischen Daten, die vorher eingelesen wurden. 
    """
    dt  = 1. / fps
    n_punkte = np.shape(x_m)[0]
    t_emp = np.linspace(0, dt*n_punkte, num=n_punkte) 
    
    # Fit in x-Richtung
    fit_param_x = np.polyfit(t_emp, x_m, 1) # fit_param: m,n mit y = m*x + n
    m = fit_param_x[0]
    n = fit_param_x[1]
    #print(m,n)
    v_x = m
    #x_0 = n
    x_0 = x_m[0]
    
    # Fit in y-Richtung
    popt, pcov = curve_fit(func, t_emp, y_m) # fit von 1/2 gt^2+b*t+c
    b = popt[0]
    c = popt[1]
    #print(b,c)
    v_y0 = b
    #y_0  = c
    y_0 = y_m[0]
    return x_0, y_0, v_x, v_y0


def func(t, b, c):
    """
    Funktion, die für den Fit benötigt wird.
    """
    g=9.81
    return -0.5*g*t*t + b*t + c


def print_Schaetzer_Anfangswerte(x_0, y_0, v_x, v_y0, rundung):
    print("geschätzte Anfangswerte der Wurfbewegung:")
    print("x_0=" + str(np.round(x_0, rundung)) + "m")
    print("y_0=" + str(np.round(y_0, rundung)) + "m")

    print("v_x0=" + str(np.round(v_x, rundung)) + "m/s")
    print("v_y0=" + str(np.round(v_y0, rundung)) + "m/s")
    return 1


##### Zelle Werte wiederherstellung ##### 
def werte_wiederherstellen():
    
    Videoanalyse_Positionen = pd.read_excel("Videoanalyse_Positionen.xlsx", index_col=0)
    Videoanalyse_Parameter  = pd.read_excel("Videoanalyse_Parameter.xlsx", index_col=0).T
    
    x = Videoanalyse_Positionen["x-Positionen in m"].values.tolist()
    y = Videoanalyse_Positionen["y-Positionen in m"].values.tolist()
    zeitpunkte = Videoanalyse_Positionen["Zeitpunkte der Bilder/Positionen in s seit erstem Bild"].values.tolist()
    
    B = float(Videoanalyse_Parameter["Bildbreite B (real) in m"].values)
    H = float(Videoanalyse_Parameter["Bildhoehe H (real) in m"].values)
    b = float(Videoanalyse_Parameter["Bildbreite B (in Pixeln)"].values)
    h = float(Videoanalyse_Parameter["Bildhoehe H (in Pixeln)"].values)
    fps = float(Videoanalyse_Parameter["fps"].values)
    delta_t = float(Videoanalyse_Parameter["delta t in s"].values)
    N_Bilder = int(Videoanalyse_Parameter["Anzahl Bilder"].values)
    N_Punkte = int(Videoanalyse_Parameter["Anzahl Punkte"].values)
    x_0 = float(Videoanalyse_Parameter["Schätzer x_0 in m"].values)
    y_0 = float(Videoanalyse_Parameter["Schätzer y_0 in m"].values)
    v_x0 = float(Videoanalyse_Parameter["Schätzer v_x0 in m/s"].values)
    v_y0 = float(Videoanalyse_Parameter["Schätzer v_y0 in m/s"].values)
                
    rundung = 2
    print_Schaetzer_Anfangswerte(x_0, y_0, v_x0, v_y0, rundung)
    
    return x, y, zeitpunkte, B, H, b, h, fps, delta_t, N_Punkte, N_Bilder, x_0, y_0, v_x0, v_y0


#### Codezelle Simulation #####

def simulation_ausführen(x_0, y_0, v_x0, v_y0, fps, N_Punkte, Nr):
    
    # Randwerte
    dt = 1. / fps # Zeit zwischen zwei Bildern in s 
    g  = 9.81 # Erdbeschleunigung in m/s^2
    a_x0 = 0.
    a_y0 = -g
    N_Punkte = int(N_Punkte) # Anzahl an Runden

    # Arrays anlegen
    a_x_array, v_x_array, x_array, a_y_array, v_y_array, y_array = tabellen_fuer_simulationsswerte_anlegen(N_Punkte, x_0, y_0, v_x0, v_y0, a_x0, a_y0)

    # Simulation durchführen:
    for r in range(1, N_Punkte, 1): # "-1", weil die Anfangswerte schon gegeben sind...
        # Einlesen der "alten" Werte (aus der vorherigen Runde) aus der Tabelle:
        a_x_alt, v_x_alt, x_alt, a_y_alt, v_y_alt, y_alt = a_x_array[r-1], v_x_array[r-1], x_array[r-1], a_y_array[r-1], v_y_array[r-1], y_array[r-1]

        # Der Spielzug, in dem wie Werte der neuen Runde nach den Spielregeln berechnet werden:
        ##### ***Trage in den nächsten Zeilen die Spielregeln für die Simulation einer Wurfbewegung ein.*** #####
        a_x_neu = 0.
        v_x_neu = v_x_alt
        x_neu = x_alt + v_x_alt*dt

        a_y_neu = -g
        v_y_neu = v_y_alt + a_y_alt * dt
        y_neu = y_alt + v_y_alt * dt

        # Und nun schreiben wir die neuen Werte in die Tabelle an der Stelle der Runde "r":
        a_x_array[r] = a_x_neu
        v_x_array[r] = v_x_neu
        x_array[r]   = x_neu

        a_y_array[r] = a_y_neu
        v_y_array[r] = v_y_neu
        y_array[r]   = y_neu
        
    print("Die Simulation wurde erfolgreich durchgeführt.")
        
    # Ergebnis als Grafik zeigen und speichern:
    df_sim = zeige_und_speichere_simulierte_werte(a_x_array, v_x_array, x_array, a_y_array, v_y_array, y_array, Nr)
    
    return df_sim


def tabellen_fuer_simulationsswerte_anlegen(N_punkte, x_0, y_0, v_x0, v_y0, a_x0, a_y0):
    """
    Diese Funktion speichert die Startwerte in der für die Simulation benötigten Syntax ab und legt die entsprechenden Arrays an. 
    """
    # Hier werden leere Arrays ("Tabellen") zum Hineinschreiben der 
    # berechneten Werte erzeugt:
    a_x_array = np.zeros([N_punkte])
    v_x_array = np.zeros([N_punkte])
    x_array = np.zeros([N_punkte])

    a_y_array = np.zeros([N_punkte])
    v_y_array = np.zeros([N_punkte])
    y_array = np.zeros([N_punkte])
        
    # Hier werden die Startwerte in die Tabelle übertragen:
    a_x_array[0] = a_x0
    v_x_array[0] = v_x0
    x_array[0] = x_0

    a_y_array[0] = a_y0
    v_y_array[0] = v_y0
    y_array[0] = y_0
    
    return a_x_array, v_x_array, x_array, a_y_array, v_y_array, y_array


def zeige_und_speichere_simulierte_werte(a_x_array, v_x_array, x_array, a_y_array, v_y_array, y_array, Nr):
    N = len(a_x_array)
    matrix_alle_werte = np.zeros([N,6])
        
    matrix_alle_werte[:,0] = a_x_array
    matrix_alle_werte[:,1] = v_x_array
    matrix_alle_werte[:,2] = x_array
    
    matrix_alle_werte[:,3] = a_y_array
    matrix_alle_werte[:,4] = v_y_array
    matrix_alle_werte[:,5] = y_array
    
    spalten = ["a_x", "v_x", "x", "a_y", "v_y", "y"]
    
    #df_sim = pd.DataFrame(data=matrix_alle_werte, index=matrix_alle_werte[:,0], columns=spalten)
    df_sim = pd.DataFrame(data=matrix_alle_werte, columns=spalten)
    #print("Hier ist der ausgefüllte Spielplan:")
    #print(df_sim)
    
    # plotten:
    fig, ax = plt.subplots()
    plt.plot(x_array, y_array, "ro")
    ax.set_title("Simulierte Flugbahn des Objekts \n Simulation mit Modell und Anfangswerten")
    ax.set_xlabel("x-Position [m]")
    ax.set_ylabel("y-Position [m]")
    plt.show()
    plt.savefig("Simulierte_Flugbahn_" + str(Nr) + ".png")
    
    DownloadButton("Simulierte_Flugbahn_" + str(Nr) + ".png")
    
    
    # speichern der Daten:
    df_sim.to_excel("Simulation" + str(Nr) + ".xlsx")    
    
    return df_sim


##### Codezelle "erzeuge Grafiken" 
def erzeuge_Grafiken(Nr):

    # Werte einlesen
    df_sim = pd.read_excel("Simulation" + str(Nr) + ".xlsx", index_col=0)
    Videoanalyse_Positionen = pd.read_excel("Videoanalyse_Positionen.xlsx", index_col=0)
    Videoanalyse_Parameter = pd.read_excel("Videoanalyse_Parameter.xlsx", index_col=0)

    x_emp = Videoanalyse_Positionen["x-Positionen in m"]
    y_emp = Videoanalyse_Positionen["y-Positionen in m"]
    x_sim = df_sim["x"]
    y_sim = df_sim["y"]

    # Plot erstellen
    fig, ax = plt.subplots() # figsize=(11.6, 8.2) für A4

    ax.plot(x_emp, y_emp, "bo", label="gemessen")
    ax.plot(x_sim, y_sim, "ro", label="simuliert")

    ax.set_ylabel('y [m]')
    ax.set_xlabel('x [m]')
    ax.set_title('Vergleich Prognose aus Simulation und Messwerte aus Beobachtung.')
    ax.legend()
    plt.savefig("Vergleich_Messwerte_Simulation_" + str(Nr) + ".png")
    plt.show()
    
    DownloadButton("Vergleich_Messwerte_Simulation_" + str(Nr) + ".png")
    
    print("Grafik erzeugt.") 
    
    
    
##### Codezelle "Erzeuge Video" #####
    
def erzeuge_Ueberlagerungsvideo(Nr):
    # Werte einlesen...
    df_sim = pd.read_excel("Simulation" + str(Nr) + ".xlsx", index_col=0)
    Videoanalyse_Positionen = pd.read_excel("Videoanalyse_Positionen.xlsx", index_col=0)
    Videoanalyse_Parameter = pd.read_excel("Videoanalyse_Parameter.xlsx", index_col=0).T

    x_emp = Videoanalyse_Positionen["x-Positionen in m"]
    y_emp = Videoanalyse_Positionen["y-Positionen in m"]
    x_sim = df_sim["x"]
    y_sim = df_sim["y"]
    
    B = float(Videoanalyse_Parameter["Bildbreite B (real) in m"].values)
    H = float(Videoanalyse_Parameter["Bildhoehe H (real) in m"].values)
    b = float(Videoanalyse_Parameter["Bildbreite B (in Pixeln)"].values)
    h = float(Videoanalyse_Parameter["Bildhoehe H (in Pixeln)"].values)
    fps = float(Videoanalyse_Parameter["fps"].values)
    delta_t = float(Videoanalyse_Parameter["delta t in s"].values)
    N_Bilder = int(Videoanalyse_Parameter["Anzahl Bilder"].values)
    N_Punkte = int(Videoanalyse_Parameter["Anzahl Punkte"].values)
    
    
    # erstelle einen Unterordner für die Überlagerungsbilder:
    try:
        os.mkdir("Ueberlagerungsbilder")
    except:
        unnuetz = 42
    
    print("Vorbereitungen für Videoerstellung abgeschlossen... (Bilder werden nun erzeugt. Das dauert einen Moment...)")

    # nun erzeugen wir zunächst die Bilder, die später in dem Überlagerungsvideo zu sehen sein sollen: 
    control = 0
    r = 0
    while control == 0:
        try:
            erstelle_ueberlagerungsplots(r, x_emp, y_emp, x_sim, y_sim, B, H)
            r = r+1
        except:
            control = 1    
    
    print(str(r) + " Einzelbilder der Überlagerungen erzeugt (Zwischenschritt).")
    

    # nun erstellen wir das neue Video daraus:
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S") # wir fügen die Zeit hinzu, damit der Dateiname immer neu ist und weder eine alte Datei überschrieben wird, noch der Fall eintritt, dass es die Datei schon gibt. 
    dateiname_video = "ueberlagerungsvideo_" + "_" + str(Nr) + "_" + current_time
    
    
    kombiniere_Bilder_zu_Video(N_Punkte, dateiname_video)
    print("Einzelbilder zu Video zusammengefügt.")

    # und nun löschen wir den Ordner das Video erzeugten Bilder wieder:
    try:
        os.rmdir("Ueberlagerungsbilder")
        nutzlos = 12
    except:
        nutzlos = 13
    
    print("Du kannst das Video mit dem Dateinamen > " + dateiname_video + ".mp4 <" + " herunterladen, wenn du auf den folgenden Button klickst. Lade das Video auf dein iPad herunter und schaut es euch einmal an.")
    
    DownloadButton(dateiname_video + '.mp4')
    
    

    

def erstelle_ueberlagerungsplots(i, x_emp, y_emp, x_sim, y_sim, B, H):                                        
    """
    Diese Funktion erstellt die Überlagerungsgrafiken als Grundlage für das spätere Video.
    """
      
    img = mpimg.imread("Ausgangsbilder" + "/frame" + str(format(i, '02d')) + ".jpg") # Realbild zum entsprechenden Schritt einlesen
        
    fig, ax = plt.subplots() # Erstellen des Überlagerungsbildes:
    
    ax.set_xlabel("x-Position [m]")
    ax.set_ylabel("y-Position [m]")
        
    # Einblenden des Realbildes
    plt.imshow(img, extent=[0, B, 0, H], animated=True) 
    
    # Einblenden der simulierten Punkte:
    plt.plot(x_sim[:i+1], y_sim[:i+1], "ro", label = "simuliert")

    # Einblenden der empirischen Werte:
    plt.plot(x_emp[:i+1], y_emp[:i+1], "bo", label = "gemessen")
    
    plt.savefig("Ueberlagerungsbilder" + "/ueberlagerungsbild" + str(format(i, '02d')) + ".png")
    plt.close() 
    
    
def kombiniere_Bilder_zu_Video(N_Punkte, dateiname_video):
    '''
    Erstellt Video aus den Überlagerungsbildern. 
    '''
    video_name = dateiname_video + '.mp4'
    images = []

    for i in range(N_Punkte):
        images.append("Ueberlagerungsbilder" + "/ueberlagerungsbild" + str(format(i, '02d')) + ".png")
                
    frame = mpimg.imread(images[0]) # hier die Bibliothek des Einlesens geändert von cv2 zu mpimg
    height, width, layers = frame.shape

    fps_new = 10 # die fps im zu erzeugenden Video

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # von https://www.programcreek.com/python/example/72134/cv2.VideoWriter
    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(video_name, fourcc, fps_new, (width,height))

    for image in images:
        video.write(cv2.imread(image)) # falls es weiterhin Probleme gibt, könnte man auch hier versuchen, ob es "imread" aus mping besser tut als cv2.

    video.release()
    









##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
#####                                                 #####
##### Funktionen aus dem zweiten Durchgang Ende 2022  ##### 
#####                                                 #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 


### Funktionen für die Komplettlösung:

def gesamtsicherung_erstellen(Gruppenname, Nr):
    
    # Einlesen der bisherigen Sicherungsdateien:
    param1 = pd.read_pickle(Gruppenname + "/s1.pkl")
    werte_exp = pd.read_pickle(Gruppenname + "/s2.pkl")
    param2 = pd.read_pickle(Gruppenname + "/s3.pkl")
    werte_sim = pd.read_pickle(Gruppenname + "/s4_sim_" + str(Nr) + ".pkl")

    # alles in ein Pandas Dateframe schreiben und als pkl und xlsx speichern:
    alles_zusammen = pd.concat([werte_exp, werte_sim, param1, param2], axis=1, ignore_index=True)
    alles_zusammen.columns = ["x-Position Ball angetippt [m]", "y-Position Ball angetippt [m]", "x-Position Ball angetippt [px]", "y-Position Ball angetippt [px]", "sim a_x [m/s^2]", "sim v_x [m/s]", "sim x [m]", "sim a_y [m/s^2]", "sim v_y [m/s]", "sim y [m]", "dateiname_realvideo", "gemessene Bildbreite B [m]", "Gruppenname", "Anzahl Bilder in Video", "fps [1/s]", "Bildhöhe [px]", "Bildbreite [px]", "Bildhöhe [m]", "Anzahl angetippte Punkte", "Schätzwert x_0 [m]", "Schätzwert y_0 [m]", "Schätzwert v_x0 [m/s]", "Schätzwert v_y0 [m/s]"]
    alles_zusammen.to_excel(Gruppenname + "/AlleWerte_" + Gruppenname + ".xlsx", sheet_name='Werte ECMB')
    alles_zusammen.to_pickle(Gruppenname + "/AlleWerte_" + Gruppenname + ".pkl")

    
def sicherungsdateien_s_loeschen(Gruppenname, Nr):
    try:
        os.remove(Gruppenname + "/s1.pkl")
        os.remove(Gruppenname + "/s2.pkl")
        os.remove(Gruppenname + "/s3.pkl")
        os.remove(Gruppenname + "/s4_sim_" + str(Nr) + ".pkl")
    except:
        print("Dateien bereits gelöscht.")

def auswertung_komplett(B, Gruppenname, x_m, y_m, N_Punkte):
    
    x_m, y_m, N_Punkte, v_x0, x_0, v_y0, y_0 = werte_wiederherstellen(Gruppenname)
    
    df_s1 = pd.read_pickle(Gruppenname + "/s1.pkl") 
    fps = df_s1["fps"]
    
    ##### ***In den nächsten Zeilen die Randwerte für die Simulation eintragen*** ####
    dt = 1. / fps # Zeit zwischen zwei Bildern in s 
    g  = 9.81 # Erdbeschleunigung in m/s^2
    ax = 0.
    ay = -g
    N = N_Punkte # Anzahl an Runden

    # *** Wenn ihr eure Simulation später überarbeitet, tragt hier eine höhere Nummer ein, um eure ersten Ergebnisse nicht zu überschreiben.
    Nr = 1

    ##### ***In den nächsten Zeilen die Anfangswerte für die Simulation eintragen*** ####
    a_x0 = ax # Startbeschleunigung in x-Richtung 
    v_x0 = v_x0 # Startgeschwindigkeit in x-Richtung
    x_0 = x_0 # Startposition x-Koordinate 

    a_y0 = ay # Startbeschleunigung in y-Richtung
    v_y0 = v_y0 # Startgeschwindigkeit in y-Richtung
    y_0 = y_0 # Startposition y-Koordinate 

    # Hier legen wir Tabellen ("arrays") für die simulierten Werte an und schreiben die Anfangswerte bereits hinein:
    a_x_array, v_x_array, x_array, a_y_array, v_y_array, y_array = tabellen_fuer_simulationsswerte_anlegen(N_Punkte, x_0, y_0, v_x0, v_y0, a_x0, a_y0)

    print("Rand- und Anfangswerte erfolgreich eingetragen und Tabellen für Spieldurchlauf angelegt.")
    
    ### Simulationsteil: 
    for r in range(1, N, 1): # "-1", weil die Anfangswerte schon gegeben sind...
        # Einlesen der "alten" Werte (aus der vorherigen Runde) aus der Tabelle:
        a_x_alt, v_x_alt, x_alt, a_y_alt, v_y_alt, y_alt = a_x_array[r-1], v_x_array[r-1], x_array[r-1], a_y_array[r-1], v_y_array[r-1], y_array[r-1]

        # Der Spielzug, in dem wie Werte der neuen Runde nach den Spielregeln berechnet werden:
        ##### ***Trage in den nächsten Zeilen die Spielregeln für die Simulation einer Wurfbewegung ein.*** #####
        a_x_neu = a_x_alt
        v_x_neu = v_x_alt + a_x_alt*dt
        x_neu = x_alt + v_x_alt*dt

        a_y_neu = a_y_alt
        v_y_neu = v_y_alt + a_y_alt * dt
        y_neu = y_alt + v_y_alt * dt

        # Und nun schreiben wir die neuen Werte in die Tabelle an der Stelle der Runde "r":
        a_x_array[r] = a_x_neu
        v_x_array[r] = v_x_neu
        x_array[r]   = x_neu

        a_y_array[r] = a_y_neu
        v_y_array[r] = v_y_neu
        y_array[r]   = y_neu
    
    # nun ist unser Spiel = unsere Simulation schon fertig! 
    print("Die Simulation ist fertig.")

    # Wir schauen uns die Werte einmal an und speichern sie gleichzeitig:
    zeige_und_speichere_simulierte_werte(a_x_array, v_x_array, x_array, a_y_array, v_y_array, y_array, Nr, Gruppenname)
    
    # Überlagerungsvideo
    erzeuge_Ueberlagerungsvideo(Nr, Gruppenname)
    
    # Grafiken
    erzeuge_Grafiken(Nr, Gruppenname)
    loesche_alle_Bilder_der_Zwischenschritte(Nr, Gruppenname)
    
    # Werte als Excel-Datei exportieren
    gesamtsicherung_erstellen(Gruppenname, Nr)
    
    
    # alle Dateien wieder löschen, die nicht gebraucht werden
    sicherungsdateien_s_loeschen(Gruppenname, Nr)
   



### Funktionen, die auf andere Weise die Anfangswerte schätzen: 

def linie_analytisch(x_0, y_0, v_x, v_y0, range_x_min, range_x_max, steps_cont):
    """
    Berechnet die analytischen Werte für den schiefen Wurf auf Basis gegebener Startwerte x_0, y_0, v_x, v_y0.
    """
    range_x = range_x_max - range_x_min
    xx = np.linspace(range_x_min, range_x_max, num=steps_cont)
    T_max = range_x / v_x
    tt = np.linspace(0, T_max, num=steps_cont)
    g = 9.81 # m/s^2
    yy = -0.5*g*tt*tt + v_y0*tt + y_0
    
    return tt, xx, yy

def berechne_startwerte_zweiframes(fps, x_m, y_m):
    """
    Berechnet die Startwerte, also Startgeschwindigkeit in x- und y-Richtung sowie die Position in x- und y-Richtung, aus den ersten beiden Punkten.
    """
    dt  = 1. / fps # Zeit zwischen zwei Bildern in s 
    
    # Startgeschwindigkeit (in x- und y-Richtung):
    v_x0 = (x_m[1] - x_m[0]) / dt
    v_y0 = (y_m[1] - y_m[0]) / dt
    
    # Startposition:
    x_0 = x_m[0]
    y_0 = y_m[0]

    print("Anfangswerte für Simulation (aus den ersten beiden Punkten geschätzt):")
    print("x_0=" + str(np.round(x_0, 4)) + "m")
    print("y_0=" + str(np.round(y_0, 4)) + "m")

    print("v_x0=" + str(np.round(v_x0, 4)) + "m/s")
    print("v_y0=" + str(np.round(v_y0, 4)) + "m/s")
    
    return v_x0, v_y0, x_0, y_0


def berechne_startwerte_max(fps, x_m, y_m):
    """
    Berechnet die Startwerte, also Startgeschwindigkeit in x- und y-Richtung sowie die Position in x- und y-Richtung, aus weiteren Punkten:
    für v_x: nutze den ersten und letzten Punkt der Datenreihe (Annahme: gleichförmige Bewegung v_x = konst)
    für v_y: nutze den Scheitelpunkt der Flugparabel (Annahme: gleichförmig beschleunigte Bewegung v_y = -g*t + v0)
    Für die Herleitung der Formel für den Scheitelpunkt s. Notizen bei Jan (TODO: wenn es klappt, das noch ordentlich dokumentieren)
    """
    # nun geht es los:
    dt  = 1. / fps # Zeit zwischen zwei Bildern in s 

    # Startposition:
    x_0 = x_m[0]
    y_0 = y_m[0]
    
    ## nun die Anfangsgeschwindigkeiten in x-Richtung:
    # zunächst bestimme die Anzahl Punkte (sollte = N_Bilder sein, aber lieber mal ausprobieren):
    n_punkte = np.shape(x_m)[0]
    # nun bestimme den letzten Punkt der Datenreihe:
    x_letzter = x_m[n_punkte-1]
    # für v_x gilt nun wegen x = v*t und damit x_letzter(nach t=n_punkte*dt) = v_x * (n*dt) folgendes: 
    v_x = (x_letzter - x_0) / (n_punkte * dt) # (Einheit: m/s)
    
    ## nun die Anfangsgeschwindigkeit in y-Richtung (Zur Herleitung der Formel s. die Notizen in Concepts):
    g = 9.81 # (m/s^2)
    # bestimme x_scheitel, den Punkt x, an dem y maximal wird. Also den Scheitelpunkt.
    index_scheitel = np.argmax(y_m)
    x_scheitel = x_m[index_scheitel]
    y_scheitel = y_m[index_scheitel]

    # nun v_y0 berechnen:
    v_y0 = (x_scheitel - x_0) * g / v_x
    
    print("Anfangswerte für Simulation (mit End- und Scheitelpunkt geschätzt):")
    print("x_0=" + str(np.round(x_0, 4)) + "m")
    print("y_0=" + str(np.round(y_0, 4)) + "m")

    print("v_x0=" + str(np.round(v_x, 4)) + "m/s")
    print("v_y0=" + str(np.round(v_y0, 4)) + "m/s")
    
    #jetzt ist der Schätzer fertig. Gib ihn zurück:
    return x_0, y_0, v_x, v_y0




def plot_alle_startwerte(fps, x_m, y_m):
    """
    Funktion erstellt einen Plot, in dem die Ergebnisse für die verschiedenen Optionen, die Startwerte zu bestimmen, dargestellt werden.
    """
    # Werte zur Bestiimmung der analytischen Trajektorie:
    n_punkte = np.shape(x_m)[0]
    x_letzter = x_m[n_punkte - 1]
    range_x_max = x_letzter
    steps_cont = n_punkte
    
    v_x0, v_y0, x_0, y_0 = berechne_startwerte_zweiframes(fps, x_m, y_m)
    range_x_min = x_0
    tt_2p, xx_2p, yy_2p = linie_analytisch(x_0, y_0, v_x0, v_y0, range_x_min, range_x_max, steps_cont)
    x_0, y_0, v_x, v_y0 = berechne_startwerte_max(fps, x_m, y_m)
    range_x_min = x_0
    tt_max, xx_max, yy_max = linie_analytisch(x_0, y_0, v_x, v_y0, range_x_min, range_x_max, steps_cont)
    x_0, y_0, v_x, v_y0 = berechne_startwerte_fit(fps, x_m, y_m)
    range_x_min = x_0
    tt_fit, xx_fit, yy_fit = linie_analytisch(x_0, y_0, v_x, v_y0, range_x_min, range_x_max, steps_cont)
    
    fig, ax = plt.subplots()
    plt.plot(x_m, y_m, marker="o", color='b', label = "empirisch")
    # die analytischen Trajektorien mit den berechneten Startwerten
    plt.plot(xx_max, yy_max, marker=".", color='black', label = "analytisch, Startwerte max.")
    plt.plot(xx_2p, yy_2p, marker=".", color='green', label = "analytisch, Startwerte 2p")
    plt.plot(xx_fit, yy_fit, marker="+", color='r', label = "analytisch, Startwerte fit")

    plt.legend()
    plt.savefig("vergleich_versch_Startwerte.png")
    plt.show()

    
   