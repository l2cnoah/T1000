# T1000
Repo for the traffic light detection


Schritt für Schritt Anleitung für Mac:

Terminal öffnen, bash einstellen

cd ~
mkdir ampel_erkennung
cd ampel_erkennung
python3 -m venv venv
source venv/bin/activate


Pakete installieren

pip install ultralytics opencv-python


SKript erstellen

touch ampel_erkennung.py
open -a "Visual Studio Code" ampel_erkennung.py 

(für "Visual Studio Code" beliebigen Editor in die Anführungszeichen eintragen)

Code einfügen, welcher im Dokument im Repo ist

Starten 

python ampel_erkennung.py



Beim ersten Start wird das Yolo Package mit heruntergeladen - dauert etwas länger, benötigt Onlineverbindung.




