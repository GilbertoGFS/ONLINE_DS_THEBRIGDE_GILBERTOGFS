import numpy as np
import random
import time

'''
COMENZAMOS CON LAS FUNCIONES
partiendo de lo que ya teniamos modifique alguna cosa y desarrole algunas otras
'''

tablero = np.full((10,10), ' ')

def crea_tablero(lado = 10):
    tablero = np.full((10,10),' ')
    return tablero

def crea_barco(tablero, barco): 
    for o in barco:
        tablero[o] = "O"
    return tablero

def disparo(tablero, apunta_dispara):
    if tablero[apunta_dispara] == "O":
        tablero[apunta_dispara] = "X"
        print("fuego")
    elif tablero[apunta_dispara] == "X":
        print("ya esta quemado esta parte del barco")
    else:
        tablero[apunta_dispara]= "s"
        print("agua movida")


#Reajuste

def restriccion_barco(tablero, barco): #En esta parte necesito saber el numero maximo de filas y columnas
    prueba_table = tablero.copy() 
    max_filas = tablero.shape[0] #lo cual accedo a ella mediante shape y le asigno una varible
    max_columnas = tablero.shape[1]
    '''En este paso me parecio mas resumido poner 2 variables asi cada vez que recorra barco, guarde los datos
    en 2 variables, esto es posible porque en barco tendre tuplas y en vez de asignar una variable aigno 2'''
    for fila, columna in barco: 
        if fila < 0 or fila >= max_filas:
            print(f"lugar ({fila}, {columna}) equivocado")
            return False
        if columna < 0 or columna >= max_columnas:
            print(f"lugar ({fila}, {columna}) equivocado")
            return False
        if tablero[fila, columna] == "O" or tablero[fila, columna] == "X":
            print(f"No puedes montar burro encima de burro ({fila}, {columna})")
            return False
        prueba_table[fila, columna] = "O"

    return prueba_table


'''En este paso quiero trabajar con longitud, la longitud sera los esloras del barco que se me piden [2,2,2,3,3,4]
como tengo una lista definida puedo trabajr con la longitud y crear una formula para que no se salga de la matriz '''

def produccion_de_buques(tablero, longitud):
    max_filas = tablero.shape[0]
    max_columnas = tablero.shape[1]
    orientacion = random.choice([0,1])
#La orientacion siempre sera 0 y 1 un numero random para saber si va vertical o horizonal y poder colocar mi barco
    if orientacion == 0:
        fila = random.randint(0,max_filas - 1)
        columna = random.randint(0, max_columnas - longitud)
    else:
        fila = random.randint(0, max_filas - longitud)
        columna = random.randint(0, max_columnas -1)
    
    barco = [] #Creo una lista vacia para poder colocar mis barco cuando recorra longitud, asi le asigno coordenadas

    for i in range(longitud):
        if orientacion == 0:
            barco.append((fila,columna + i))
        else:
            barco.append((fila + i, columna))

    return barco


def colocar_barcos(tablero): # Establezco el size para los tamaños de los barcos
    size = [2,2,2,3,3,4]
    barcos_colocados = []   
    for longitud in size:
        colocado = False # El false lo uso para poder entrar en el bucle while 

        while not colocado: # llamo a mis funciones para colocar mis barcos y aparte que no se salga de la tabla
            barco = produccion_de_buques(tablero, longitud)
            resultado = restriccion_barco(tablero,barco)

            if resultado is not False: 
                tablero = crea_barco(tablero,barco)
                barcos_colocados.append(barco)
                colocado = True # cambio colocado a True para poder salir del bucle while
    return tablero, barcos_colocados


def juego():
    tablero = crea_tablero()
    tablero = colocar_barcos(tablero)

    terminado = False # Vuelvo a colocar false para entrar en el puble

    while not terminado: # Necesitamos crear un input para poder colocar las coordenadas
        fila = int(input("para fila del 0 al 9 cual te gusta?"))
        columna = int(input("del 0 al 9 a donde quieres darle"))
        disparo(tablero, (fila, columna))
        print(tablero)

        time.sleep(3) # Uso la libreria time y el metodo sleep para dar unos segundos para empezar el bucle de la maquina

        machine_fila = random.randint(0,9) # Como es la maquina las coordenadas las hacemos random
        machine_columna = random.randint(0,9)
        print(f"pium pium ({machine_fila}, {machine_columna})")
        disparo(tablero, (machine_fila, machine_columna))
        print(tablero)


"""
En esta funcion quiero recorrer el tablero para saber si hay X o no
para luego usarla mas adelante para saber si ya todos los barcos estan hundidos

"""

def hay_flota_o_no(barco, tablero): 
    for fila, columna in barco:
        if tablero[fila, columna] != "X":
            return False
    return True    


def todos_hundidos(barcos, tablero):
    for barco in barcos:
        if not hay_flota_o_no(barco, tablero):
            return False
    return True