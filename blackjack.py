"""
El problema del blackjack simplificado como un problema de aprendizaje por refuerzo

"""

from RL import MDPsim, SARSA, Q_learning, PoliticaGreedy
from random import random, randint

class BlackJack(MDPsim):
    """
    Clase que representa un MDP para el problema del jugador.
    
    El jugador tiene un capital inicial y el objetivo es llegar a un capital
    objetivo o quedarse sin dinero.
    
    """
    def __init__(self, gama):
        # Definir los estados del blackjack: tuplas (suma_jugador, carta_visible_crupier, as_usable)
        # Suma del jugador: de 4 a 21
        # Carta visible del crupier: de 1 a 10 (donde 1 representa al As)
        # As usable: True, False
        # Se añade un estado especial 'terminal' para simplificar la lógica de fin del juego
        self.estados = []
        for suma in range(4, 22):
            for visible in range(1, 11):
                for as_usable in (True, False):
                    self.estados.append((suma, visible, as_usable))
        self.estados.append('terminal')
        self.estados = tuple(self.estados)
        self.gama = gama

    def reparte_carta(self):
        # Simular baraja infinita:
        # 13 rangos posibles (As, 2, ..., 10, J, Q, K) con prob 1/13 cada uno.
        # Las figuras J, Q, K valen 10. Por tanto, el valor 10 tiene probabilidad 4/13.
        # El As se representa como 1.
        carta = randint(1, 13)
        return 10 if carta >= 10 else carta

    def calcular_mano(self, mano):
        # Calcula la suma total de una mano y si tiene un As usable.
        # Mano es una lista de valores de cartas (donde 1 representa al As).
        suma = sum(mano)
        as_usable = False
        if 1 in mano and suma + 10 <= 21:
            suma += 10
            as_usable = True
        return suma, as_usable
        
    def estado_inicial(self):
        # Repartir 2 cartas al jugador y 2 al crupier (una visible, una oculta)
        self.mano_jugador = [self.reparte_carta(), self.reparte_carta()]
        self.mano_crupier = [self.reparte_carta(), self.reparte_carta()]
        
        # Calcular la suma inicial y As usable del jugador
        suma_jugador, as_usable = self.calcular_mano(self.mano_jugador)
        carta_visible_crupier = self.mano_crupier[0]
        
        # Guardar bandera por si hay Blackjack natural inmediato del jugador
        self.natural_blackjack = (suma_jugador == 21)
        
        return (suma_jugador, carta_visible_crupier, as_usable)
    
    def acciones_legales(self, s):
        # Si el estado es terminal, no hay acciones legales
        if self.es_terminal(s):
            return []
        # Acciones: 0 = Stand (Plantarse), 1 = Hit (Pedir)
        return [0, 1]
    
    def recompensa(self, s, a, s_):
        # La recompensa en cada estado no terminal es 0
        if not self.es_terminal(s_):
            return 0
        
        # Al llegar a un estado terminal, calculamos la recompensa final
        suma_jugador, _ = self.calcular_mano(self.mano_jugador)
        
        # Caso 1: El jugador se pasó de 21 (Bust)
        if suma_jugador > 21:
            return -1
            
        # Caso 2: El jugador se plantó (Stand) o hubo Blackjack natural
        suma_crupier, _ = self.calcular_mano(self.mano_crupier)
        
        # Si el jugador tiene Blackjack natural (21 con sus 2 primeras cartas)
        if len(self.mano_jugador) == 2 and suma_jugador == 21:
            # Comprobar si el crupier también tiene Blackjack natural con sus 2 primeras cartas
            suma_crupier_inicial, _ = self.calcular_mano(self.mano_crupier[:2])
            if suma_crupier_inicial == 21:
                return 0 # Empate (ambos tienen Blackjack natural)
            else:
                return 1.5 # Victoria por Blackjack natural (paga 3 a 2)
                
        # Comparación estándar tras plantarse
        if suma_crupier > 21:
            return 1 # El crupier se pasó de 21, gana el jugador
        elif suma_jugador > suma_crupier:
            return 1 # Jugador tiene más puntos, gana el jugador
        elif suma_jugador < suma_crupier:
            return -1 # Crupier tiene más puntos, gana el crupier
        else:
            return 0 # Empate (Push)
    
    def transicion(self, s, a):
        # Si ya estamos en estado terminal, no hay cambio
        if self.es_terminal(s):
            return 'terminal'
        
        suma_jugador, carta_visible_crupier, as_usable = s
        
        if a == 1:  # Hit (Pedir carta)
            self.mano_jugador.append(self.reparte_carta())
            nueva_suma, nuevo_as_usable = self.calcular_mano(self.mano_jugador)
            
            # Si se pasa de 21, termina el juego (Bust)
            if nueva_suma > 21:
                return 'terminal'
            return (nueva_suma, carta_visible_crupier, nuevo_as_usable)
            
        else:  # Stand (Plantarse)
            # Turno del crupier: pide cartas hasta tener al menos 17
            suma_crupier, _ = self.calcular_mano(self.mano_crupier)
            while suma_crupier < 17:
                self.mano_crupier.append(self.reparte_carta())
                suma_crupier, _ = self.calcular_mano(self.mano_crupier)
            
            # El juego termina después de plantarse el jugador
            return 'terminal'
    
    def es_terminal(self, s):
        # El juego termina si el estado es el especial 'terminal'
        # o si la suma del jugador supera 21 (Bust)
        if s == 'terminal':
            return True
        if isinstance(s, tuple) and s[0] > 21:
            return True
        return False


if __name__ == "__main__":

    blackjack = BlackJack(gama=1,...) # TODO: agregar los parámetros necesarios para el blackjack   

    # TODO: definir los parámetros de SARSA y Q-learning, luego crear las instancias 
    # de cada algoritmo
    Q_sarsa = SARSA( blackjack, alfa=..., epsilon=..., n_ep=..., n_iter=...)
    Q_learning = Q_learning( blackjack, alfa=..., epsilon=..., n_ep=..., n_iter=...)

    # Encuentra las políticas óptimas para cada algoritmo
    pi_s = PoliticaGreedy(Q_sarsa)
    pi_q = PoliticaGreedy(Q_learning)

    # Imprime las políticas óptimas para cada estado no terminal
    print("Estado".center(10) + '|' +  "SARSA".center(10) + '|' + "Q-learning".center(10))
    print("-"*10 + '|' + "-"*10 + '|' + "-"*10)
    for s in blackjack.estados:
        if not blackjack.es_terminal(s):
            print(str(s).center(10) + '|' 
                  + str(pi_s(s)).center(10) + '|' 
                  + str(pi_q(s)).center(10))
    print("-"*10 + '|' + "-"*10 + '|' + "-"*10)


"""
****************************************************************************************
Responde las siguientes preguntas:

1. ¿Cuáles son los estados, acciones, recompensas y transiciones en el problema del 
    blackjack?  

2. ¿Cómo se pueden representar los estados del blackjack de manera eficiente para el 
    aprendizaje por refuerzo?

3. ¿Qué pasa si se modifica el valor de epsilón de la política epsilon-greedy?

4. ¿Cómo afecta el valor de alfa en la convergencia de los algoritmos SARSA y Q-learning?

5. ¿Cuál de los dos algoritmos, SARSA o Q-learning, consideras que es más adecuado para 
   el problema del blackjack y por qué?

6. ¿Se puede explicar con cierta lógica del juego la política óptima encontrada por cada 
   algoritmo? ¿Qué acciones se toman en cada estado y por qué?
****************************************************************************************
"""
