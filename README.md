# B.Eng-Thesis-WaSoR
My engineering thesis project - A system for visual sorting plastic waste. I had used 2 machine learning algorithms to recognize between different types of plastic waste.

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Dataset](#dataset)
* [SVM](#svm)
* [CNN](#cnn)
* [Project summary](#project-summary)
* [Contact](#contact)

## General info
In the recent years, several robotics companies(like [@ZenRobotics](https://zenrobotics.com)) have developed a new system for waste management - Waste Sorting Robots(hence *WaSoR*). Combining robotic arms, computer vision and machine learning to segregate waste moving on a conveyor belt. 

This project is my take on creating such application, starting from the crucial part - properly recognizing types of waste. My goal was to create a machine learning algorithm that distinguishes between different types of plastic trash.
I had used an international classification of:
LDPE, HDPE, PET, PP, PS, Other 


- Standard classes: 
- Later on I'd proposed: LDPE, HDPE, PETb, Misc

I had used Support Vector Machines, with Bag of Features technique, and Convolutional Neural Network with Transfer Learning.

1. Maszyna wektorów nośnych z techniką bag of features - sprawdzony algorytm, który
uzyskał średnio 60% dokładności. w szczególności skuteczny przy prostych, jednolitych, klasach.
2. Konwolucyjna sieć neuronowa, z wykorzystaniem metody transfer learning - nowoczesne podejście, które eksperymentalnie dochodzi do 90% dokładności. Wymagający, ale bardziej obiecujący klasyfikator.



In regard to images I'm using [@WaDaBa](http://wadaba.pcz.pl/#home) database(in a separate directory), plus images made by myself.


## Technologies
* Python 3.0
* Keras
* Matlab - R2017b
* Excel

## Dataset

![Example screenshot](./img/screenshot.png)

## SVM
Wzorując się na pracach SAKR, YANG, najpierw zrealizowałem algorytm Support Vector Machine, z techniką Bag of Features. Opracowałem go w Matlabie, korzystając z Machine Learning Toolbox

Główny skrypt znajduje się tu.

Na podstawowej bazie Plasor udało sie uzyskać
![Example screenshot](./img/screenshot.png)

Zaś przy próbie optymalizacji klas:
![Example screenshot](./img/screenshot.png)

## CNN
Ze względu na mały dataset, niektóre klasy po kilkadziesiąt zdjęć, wzorowałem się na artykułach krozsytających z Transfer Learning, jak Cumerwerwe, Xianjio. 

Na podstawowej bazie Plasor udało sie uzyskać
![Example screenshot](./img/screenshot.png)

Zaś przy próbie optymalizacji klas:
![Example screenshot](./img/screenshot.png)

## Project summary
Sortowanie odpadów z tworzyw sztucznych jest dość trudnym zagadnieniem. Obiekty potrafią zarówno być różnorodne w ramach jednego rodzaju, jak i występują podobieństwa pomiędzy klasami. Każdy algorytm, z różnymi parametrami, radzi sobie lepiej lub gorzej, w danych warunkach.

Pomysły jak możnaby polepszyć rozpoznawalność
1. Zmiana definicji klas, stworzenie grup jak np. „kosmetyki”, „folie”.
2. Zastosowanie fuzji czujników. Połączenie obrazu z kamer z analizą spektralną, czy
sensorami taktylnymi.
3. Połączenie różnych klasyfikatorów wyspecjalizowanych w innych klasach.


## Contact
Created by [@Filip Adamcewicz](https://www.linkedin.com/in/filippo011235/) - fadamcewicz1@gmail.com - feel free to contact me! 
