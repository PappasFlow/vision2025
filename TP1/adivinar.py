# Trabajo práctico número 0

import random as rm

def adivina():
    intentos = 10
    numero = rm.randint(0, 10)
    while intentos > 0:
        print('Le quedan ' + str(intentos) + ' intentos')
        try:
            n = int(input('Ingrese un número: '))
            if numero == n:
                print('¡Ganaste!')
                break
            elif n > numero:
                print('El número es menor')
            else:
                print('El número es mayor')
            intentos -= 1
        except ValueError:
            print('Por favor, ingrese un número válido.')
    else:
        print('Se acabaron los intentos, el número era ' + str(numero))


print('El juego es adivinar un número')
adivina()
