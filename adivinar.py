#Trabajo pratico numero 0

import random as rm

def adivina():
	intentos = 10
	numero = rm.randint(0,10)
	while(intentos>0):
		print('Le quedan ' + str(intentos) + ' intentos')
		n = int(input('Ingrese un numero: '))
		if(numero == n):
			print('Ganaste')
			break
		elif (n>numero):
			print('El numero es menor')
		else:
			print('El numero es mayor')
		intentos -=1
	else:
		print('se acabaron los intentos, el numero era '+ str(numero))


print('El juego es adivinar un numero')
adivina()
